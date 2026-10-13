from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BODY_COLORS = (1, 2, 3, 4)
TAIL_COLORS = (7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")

# Base body templates are right-facing: the tail attachment lies at the midpoint
# of the right face. Every template is 4-connected, uses its full bbox, and is
# not symmetric under left-right reflection.
BASE_RIGHT_TEMPLATES = [
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=int,
    ),
]

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 80


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_patch(coords: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(color)
    return patch, (r0, c0, r1, c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = 0) -> None:
    h, w = patch.shape
    rr1 = r0 + h
    cc1 = c0 + w
    if r0 < 0 or c0 < 0 or rr1 > dst.shape[0] or cc1 > dst.shape[1]:
        raise AssertionError("patch out of bounds")
    if transparent == 0:
        mask = patch != 0
    else:
        mask = patch != int(transparent)
    view = dst[r0:rr1, c0:cc1]
    view[mask] = patch[mask]


def _orientation_from_tail(body_mask: np.ndarray, tail_coords: np.ndarray) -> tuple[str, int, tuple[int, int]]:
    rows = sorted(set(map(int, tail_coords[:, 0])))
    cols = sorted(set(map(int, tail_coords[:, 1])))
    if len(rows) == 1:
        row = rows[0]
        cmin, cmax = cols[0], cols[-1]
        assert cols == list(range(cmin, cmax + 1)), "tail must be a contiguous line"
        left_end = (row, cmin)
        right_end = (row, cmax)
        left_touch = any(
            0 <= nr < body_mask.shape[0] and 0 <= nc < body_mask.shape[1] and body_mask[nr, nc]
            for nr, nc in _neighbors4(*left_end)
        )
        right_touch = any(
            0 <= nr < body_mask.shape[0] and 0 <= nc < body_mask.shape[1] and body_mask[nr, nc]
            for nr, nc in _neighbors4(*right_end)
        )
        assert left_touch ^ right_touch, "tail must touch the body at exactly one end"
        return ("right", len(cols), left_end) if left_touch else ("left", len(cols), right_end)

    if len(cols) == 1:
        col = cols[0]
        rmin, rmax = rows[0], rows[-1]
        assert rows == list(range(rmin, rmax + 1)), "tail must be a contiguous line"
        top_end = (rmin, col)
        bottom_end = (rmax, col)
        top_touch = any(
            0 <= nr < body_mask.shape[0] and 0 <= nc < body_mask.shape[1] and body_mask[nr, nc]
            for nr, nc in _neighbors4(*top_end)
        )
        bottom_touch = any(
            0 <= nr < body_mask.shape[0] and 0 <= nc < body_mask.shape[1] and body_mask[nr, nc]
            for nr, nc in _neighbors4(*bottom_end)
        )
        assert top_touch ^ bottom_touch, "tail must touch the body at exactly one end"
        return ("down", len(rows), top_end) if top_touch else ("up", len(rows), bottom_end)

    raise AssertionError("tail must be horizontal or vertical")


def _analyze_sprite(grid: np.ndarray, coords: np.ndarray) -> dict:
    sprite_r0, sprite_c0, sprite_r1, sprite_c1 = _bbox(coords)
    sprite = grid[sprite_r0 : sprite_r1 + 1, sprite_c0 : sprite_c1 + 1]
    local_coords = coords - np.array([sprite_r0, sprite_c0], dtype=int)
    colors = sorted(set(map(int, sprite[local_coords[:, 0], local_coords[:, 1]])) - {BLACK})
    assert len(colors) == 2, "each sprite must use one body color and one tail color"

    per_color: list[tuple[int, np.ndarray]] = []
    for color in colors:
        parts = _components_from_mask(sprite == color)
        assert len(parts) == 1, "body and tail colors must each form one component inside the sprite"
        per_color.append((color, parts[0]))

    per_color.sort(key=lambda item: item[1].shape[0])
    tail_color, tail_local = per_color[0]
    body_color, body_local = per_color[1]
    assert body_local.shape[0] > tail_local.shape[0], "body must be the larger component"

    body_coords = body_local + np.array([sprite_r0, sprite_c0], dtype=int)
    tail_coords = tail_local + np.array([sprite_r0, sprite_c0], dtype=int)
    assert body_coords.shape[0] > tail_coords.shape[0], "body must be the larger component"

    body_patch, body_local_bbox = _crop_patch(body_local, body_color)
    body_mask = body_patch == body_color
    direction, length, near_global = _orientation_from_tail(
        body_mask=np.pad(body_mask, 1),
        tail_coords=tail_local - np.array([body_local_bbox[0] - 1, body_local_bbox[1] - 1], dtype=int),
    )

    body_bbox = (
        sprite_r0 + body_local_bbox[0],
        sprite_c0 + body_local_bbox[1],
        sprite_r0 + body_local_bbox[2],
        sprite_c0 + body_local_bbox[3],
    )
    r0, c0, r1, c1 = body_bbox
    h, w = body_patch.shape
    if direction == "right":
        assert near_global == (h // 2 + 1, w + 1), "tail must attach at the right midpoint"
    elif direction == "left":
        assert near_global == (h // 2 + 1, 0), "tail must attach at the left midpoint"
    elif direction == "down":
        assert near_global == (h + 1, w // 2 + 1), "tail must attach at the bottom midpoint"
    else:
        assert near_global == (0, w // 2 + 1), "tail must attach at the top midpoint"

    assert 2 <= length <= 4, "tail length must be between 2 and 4"
    assert 4 <= int(body_mask.sum()) <= 8, "body size must be between 4 and 8"

    mirrored = np.fliplr(body_patch) if direction in ("left", "right") else np.flipud(body_patch)
    assert not np.array_equal(mirrored != 0, body_mask), "body must not be symmetric about the reflection axis"

    if direction == "right":
        copy_r0, copy_c0 = r0, c1 + length + 1
    elif direction == "left":
        copy_r0, copy_c0 = r0, c0 - length - w
    elif direction == "down":
        copy_r0, copy_c0 = r1 + length + 1, c0
    else:
        copy_r0, copy_c0 = r0 - length - h, c0

    return {
        "body_color": int(body_color),
        "tail_color": int(tail_color),
        "body_coords": body_coords,
        "tail_coords": tail_coords,
        "body_patch": body_patch,
        "body_bbox": body_bbox,
        "direction": direction,
        "length": int(length),
        "copy_top_left": (int(copy_r0), int(copy_c0)),
    }


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    sprites = [_analyze_sprite(g, comp) for comp in _components_from_mask(g != BLACK)]
    assert 2 <= len(sprites) <= 3, "expected 2 or 3 sprites"

    out = g.copy()
    for sprite in sprites:
        out[sprite["tail_coords"][:, 0], sprite["tail_coords"][:, 1]] = BLACK

    for sprite in sprites:
        body_patch = sprite["body_patch"]
        mirrored = np.fliplr(body_patch) if sprite["direction"] in ("left", "right") else np.flipud(body_patch)
        mirrored = np.where(mirrored != 0, sprite["tail_color"], 0).astype(int)
        cr0, cc0 = sprite["copy_top_left"]
        h, w = mirrored.shape
        assert 0 <= cr0 and 0 <= cc0 and cr0 + h <= g.shape[0] and cc0 + w <= g.shape[1], "reflected copy must stay in bounds"
        target = out[cr0 : cr0 + h, cc0 : cc0 + w]
        assert not np.any((mirrored != 0) & (target != BLACK)), "reflected copy must not overlap existing content"
        _paste(out, mirrored, cr0, cc0)

    return out, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored tail with an equally spaced reflected copy of its body recolored by the tail.

    Input structure:
    - The grid has a black background and contains 2 or 3 disconnected two-color sprites.
    - Within each sprite, the larger monochrome component is a small non-symmetric body and the smaller one is a straight one-pixel-wide tail.
    - Every tail extends from the midpoint of one body face in one cardinal direction.

    Rule:
    - For each sprite, read the tail direction and its length.
    - Erase the tail to black.
    - Mirror the body across the axis perpendicular to the tail direction: left/right tails use a left-right mirror, up/down tails use an up-down mirror.
    - Place that mirrored copy on the tail side so the empty gap between the body face and the copy's nearest face equals the tail length, and recolor the copy with the tail color.
    - Keep the original body unchanged.

    Key insight:
    - The tail is not painted into the output; its orientation selects the reflection axis and its length becomes the spacing between the original body and the recolored mirror copy.
    """
    return _solve(grid)[0]


def _orient_template(template: np.ndarray, direction: str) -> np.ndarray:
    if direction == "right":
        return template.copy()
    if direction == "left":
        return np.fliplr(template)
    if direction == "down":
        return np.rot90(template, k=3)
    if direction == "up":
        return np.rot90(template, k=1)
    raise ValueError(direction)


def _sample_directions(n: int) -> list[str]:
    if n == 2:
        dirs = [random.choice(("left", "right")), random.choice(("up", "down"))]
        random.shuffle(dirs)
        return dirs
    return random.sample(DIR_ORDER, 3)


def _sample_lengths(n: int) -> list[int]:
    while True:
        lengths = [random.randint(2, 4) for _ in range(n)]
        if len(set(lengths)) >= 2:
            return lengths


def _choose_template(direction: str, length: int) -> np.ndarray:
    choices = []
    for base in BASE_RIGHT_TEMPLATES:
        patch = _orient_template(base, direction)
        if int(patch.sum()) <= length:
            continue
        mirrored = np.fliplr(patch) if direction in ("left", "right") else np.flipud(patch)
        if np.array_equal(mirrored, patch):
            continue
        choices.append(patch)
    if not choices:
        raise RuntimeError("no valid body template")
    return random.choice(choices).copy()


def _copy_local_coords(body_patch: np.ndarray, direction: str, length: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = body_patch.shape
    body_coords = [tuple(map(int, rc)) for rc in np.argwhere(body_patch != 0)]
    if direction == "right":
        attach = (h // 2, w - 1)
        copy_shift = (0, w + length)
    elif direction == "left":
        attach = (h // 2, 0)
        copy_shift = (0, -w - length)
    elif direction == "down":
        attach = (h - 1, w // 2)
        copy_shift = (h + length, 0)
    else:
        attach = (0, w // 2)
        copy_shift = (-h - length, 0)
    dr, dc = DIRS[direction]
    tail_coords = [(attach[0] + dr * k, attach[1] + dc * k) for k in range(1, length + 1)]
    mirrored = np.fliplr(body_patch) if direction in ("left", "right") else np.flipud(body_patch)
    copy_coords = [(copy_shift[0] + int(r), copy_shift[1] + int(c)) for r, c in np.argwhere(mirrored != 0)]
    return body_coords, tail_coords, copy_coords


def _dilate_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _render_input(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0 = spec["top_left"]
        body_patch = np.where(spec["body_patch"] != 0, spec["body_color"], 0).astype(int)
        _paste(grid, body_patch, r0, c0)
        for tr, tc in spec["tail_cells"]:
            grid[tr, tc] = spec["tail_color"]
    return grid


def _render_output(h: int, w: int, specs: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0 = spec["top_left"]
        body_patch = np.where(spec["body_patch"] != 0, spec["body_color"], 0).astype(int)
        _paste(out, body_patch, r0, c0)
        mirrored = np.fliplr(spec["body_patch"]) if spec["direction"] in ("left", "right") else np.flipud(spec["body_patch"])
        mirrored = np.where(mirrored != 0, spec["tail_color"], 0).astype(int)
        cr0, cc0 = spec["copy_top_left"]
        _paste(out, mirrored, cr0, cc0)
    return out


def _render_no_mirror_output(h: int, w: int, specs: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0 = spec["top_left"]
        body_patch = np.where(spec["body_patch"] != 0, spec["body_color"], 0).astype(int)
        _paste(out, body_patch, r0, c0)
        copy_patch = np.where(spec["body_patch"] != 0, spec["tail_color"], 0).astype(int)
        cr0, cc0 = spec["copy_top_left"]
        _paste(out, copy_patch, cr0, cc0)
    return out


def _render_fixed_gap_output(h: int, w: int, specs: list[dict], fixed_gap: int = 2) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0 = spec["top_left"]
        body_patch = np.where(spec["body_patch"] != 0, spec["body_color"], 0).astype(int)
        _paste(out, body_patch, r0, c0)
        mirrored = np.fliplr(spec["body_patch"]) if spec["direction"] in ("left", "right") else np.flipud(spec["body_patch"])
        mirrored = np.where(mirrored != 0, spec["tail_color"], 0).astype(int)
        bh, bw = spec["body_patch"].shape
        if spec["direction"] == "right":
            cr0, cc0 = r0, c0 + bw + fixed_gap
        elif spec["direction"] == "left":
            cr0, cc0 = r0, c0 - bw - fixed_gap
        elif spec["direction"] == "down":
            cr0, cc0 = r0 + bh + fixed_gap, c0
        else:
            cr0, cc0 = r0 - bh - fixed_gap, c0
        _paste(out, mirrored, cr0, cc0)
    return out


def _place_specs(h: int, w: int, specs: list[dict]) -> bool:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]["all_local_cells"]), reverse=True)
    bounds = []
    for idx in order:
        spec = specs[idx]
        rows = [r for r, _ in spec["all_local_cells"]]
        cols = [c for _, c in spec["all_local_cells"]]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        bounds.append((idx, min_r, max_r, min_c, max_c))

    for _ in range(PLACE_ATTEMPTS):
        reserved: set[tuple[int, int]] = set()
        for spec in specs:
            spec.pop("top_left", None)
            spec.pop("tail_cells", None)
            spec.pop("copy_top_left", None)

        success = True
        for idx, min_r, max_r, min_c, max_c in bounds:
            spec = specs[idx]
            placed = False
            for _ in range(160):
                r0 = random.randint(-min_r, h - max_r - 1)
                c0 = random.randint(-min_c, w - max_c - 1)
                all_cells = {(r0 + r, c0 + c) for r, c in spec["all_local_cells"]}
                dilated = _dilate_cells(all_cells, h, w)
                if dilated & reserved:
                    continue

                spec["top_left"] = (r0, c0)
                spec["tail_cells"] = [(r0 + r, c0 + c) for r, c in spec["tail_local"]]
                spec["copy_top_left"] = (r0 + spec["copy_shift"][0], c0 + spec["copy_shift"][1])
                reserved.update(dilated)
                placed = True
                break

            if not placed:
                success = False
                break

        if success:
            return True

    return False


def _build_specs(n: int) -> list[dict]:
    directions = _sample_directions(n)
    lengths = _sample_lengths(n)
    body_colors = random.sample(BODY_COLORS, n)
    tail_colors = random.sample(TAIL_COLORS, n)
    specs: list[dict] = []
    for direction, length, body_color, tail_color in zip(directions, lengths, body_colors, tail_colors):
        patch = _choose_template(direction, length)
        body_local, tail_local, copy_local = _copy_local_coords(patch, direction, length)
        rows = [r for r, _ in copy_local + tail_local + body_local]
        cols = [c for _, c in copy_local + tail_local + body_local]
        if direction == "right":
            copy_shift = (0, patch.shape[1] + length)
        elif direction == "left":
            copy_shift = (0, -patch.shape[1] - length)
        elif direction == "down":
            copy_shift = (patch.shape[0] + length, 0)
        else:
            copy_shift = (-patch.shape[0] - length, 0)
        specs.append(
            {
                "direction": direction,
                "length": length,
                "body_color": body_color,
                "tail_color": tail_color,
                "body_patch": patch,
                "body_local": body_local,
                "tail_local": tail_local,
                "copy_shift": copy_shift,
                "all_local_cells": set(body_local + tail_local + copy_local),
                "local_span": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return specs


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    try:
        actual, analyzed = _solve(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(analyzed) != len(specs):
        return False

    truth = Counter((s["body_color"], s["tail_color"], s["direction"], s["length"]) for s in specs)
    recovered = Counter((s["body_color"], s["tail_color"], s["direction"], s["length"]) for s in analyzed)
    if truth != recovered:
        return False

    if len({s["length"] for s in specs}) < 2:
        return False
    if {s["direction"] in ("left", "right") for s in specs} != {False, True}:
        return False

    if np.array_equal(expected, _render_no_mirror_output(*inp.shape, specs)):
        return False
    try:
        fixed_gap = _render_fixed_gap_output(*inp.shape, specs, fixed_gap=2)
    except AssertionError:
        fixed_gap = None
    if fixed_gap is not None and np.array_equal(expected, fixed_gap):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choices([2, 3], weights=[2, 3])[0]
    specs = _build_specs(n)

    for _ in range(PLACE_ATTEMPTS):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        placed = [dict(spec) for spec in specs]
        if not _place_specs(h, w, placed):
            continue
        inp = _render_input(h, w, placed)
        expected = _render_output(h, w, placed)
        if _semantic_checks(inp, expected, placed):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tail-reflection puzzle input")
