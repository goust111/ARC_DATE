import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
FRAME_COLORS = {RED, BLUE, GREEN}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[top + r, left + c] = color


def _mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _shape_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _random_connected_mask(canvas_h: int, canvas_w: int, area: int) -> np.ndarray:
    start = (random.randrange(canvas_h), random.randrange(canvas_w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    raw = np.zeros((canvas_h, canvas_w), dtype=bool)
    for r, c in cells:
        raw[r, c] = True
    return _crop_mask(raw)


def _sample_shape(predicate, canvas_choices: list[tuple[int, int]]) -> np.ndarray:
    for _ in range(400):
        h, w = random.choice(canvas_choices)
        area = random.randint(3, min(6, h * w))
        mask = _random_connected_mask(h, w, area)
        if predicate(mask):
            return mask
    raise RuntimeError("failed to sample a valid shape")


def _rotate_cw_about_center(mask: np.ndarray, top: int, left: int) -> tuple[np.ndarray, int, int]:
    rotated = np.rot90(mask, -1)
    h, w = mask.shape
    new_h, new_w = rotated.shape
    dr = h - new_h
    dc = w - new_w
    assert dr % 2 == 0 and dc % 2 == 0
    return rotated, top + dr // 2, left + dc // 2


def _adjacent_to_marker(cells: list[tuple[int, int]], marker: tuple[int, int]) -> bool:
    mr, mc = marker
    return any(abs(r - mr) + abs(c - mc) == 1 for r, c in cells)


def _direction_from_marker(
    marker: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int]]:
    yr, yc = marker
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[str, tuple[int, int]]] = []
    if r0 <= yr <= r1 and yc < c0:
        candidates.append(("left", (0, -1)))
    if r0 <= yr <= r1 and yc > c1:
        candidates.append(("right", (0, 1)))
    if c0 <= yc <= c1 and yr < r0:
        candidates.append(("up", (-1, 0)))
    if c0 <= yc <= c1 and yr > r1:
        candidates.append(("down", (1, 0)))
    if len(candidates) == 1:
        return candidates[0]

    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    fallback: list[tuple[str, tuple[int, int], float]] = []
    if r0 <= yr <= r1 and yc < center_c:
        fallback.append(("left", (0, -1), center_c - yc))
    if r0 <= yr <= r1 and yc > center_c:
        fallback.append(("right", (0, 1), yc - center_c))
    if c0 <= yc <= c1 and yr < center_r:
        fallback.append(("up", (-1, 0), center_r - yr))
    if c0 <= yc <= c1 and yr > center_r:
        fallback.append(("down", (1, 0), yr - center_r))
    assert len(fallback) == 1
    name, step, _ = fallback[0]
    return name, step


def _find_containers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    containers = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in FRAME_COLORS or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _shape_bbox(cells)
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            expected = {
                (rr, cc)
                for rr in range(r0, r1 + 1)
                for cc in range(c0, c1 + 1)
                if rr in (r0, r1) or cc in (c0, c1)
            }
            assert box_h >= 3 and box_w >= 3
            assert set(cells) == expected
            assert r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            assert np.isin(interior, [BG, GREY, YELLOW]).all()
            yellow_rel = np.argwhere(interior == YELLOW)
            grey_rel = np.argwhere(interior == GREY)
            assert len(yellow_rel) == 1
            assert 3 <= len(grey_rel) <= 6

            grey_mask = interior == GREY
            comps = _components(grey_mask)
            assert len(comps) == 1
            grey_cells = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in grey_rel]
            gr0, gc0, gr1, gc1 = _shape_bbox(grey_cells)
            grey_bbox_mask = g[gr0 : gr1 + 1, gc0 : gc1 + 1] == GREY

            containers.append(
                {
                    "color": color,
                    "frame_bbox": (r0, c0, r1, c1),
                    "grey_bbox": (gr0, gc0, gr1, gc1),
                    "grey_mask": grey_bbox_mask,
                    "grey_cells": grey_cells,
                    "yellow": (r0 + 1 + int(yellow_rel[0, 0]), c0 + 1 + int(yellow_rel[0, 1])),
                }
            )
    containers.sort(key=lambda item: item["frame_bbox"])
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each colored frame as a container type and transform the grey object inside it.

    Input structure:
    - The background is black.
    - Several non-overlapping 1-pixel-thick rectangular frames appear away from the grid edge; frame colors are red, blue, or green.
    - Each frame interior contains one 4-connected grey shape of area 3-6 and exactly one yellow marker pixel.

    Rule:
    - Red frame: mirror the grey shape in place inside its own bounding box. If the yellow marker sits to the left or right of that box while sharing its row band, mirror left-right; if it sits above or below while sharing the box's column band, mirror up-down. Remove the yellow marker.
    - Blue frame: rotate the grey shape 90 degrees clockwise around the center of its own bounding box and redraw it in place. Remove the yellow marker.
    - Green frame: translate the grey shape step by step toward the yellow marker until some grey cell becomes 4-neighbor adjacent to the marker. Keep the yellow marker.
    - All frame borders stay unchanged and the output grid keeps the same size.

    Key insight:
    - Frame color selects the transformation family, while the same yellow marker has different meanings in different frames: mirror-axis disambiguation for red, irrelevant distractor for blue, and movement target for green.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for container in _find_containers(g):
        color = int(container["color"])
        fr0, fc0, fr1, fc1 = container["frame_bbox"]
        gr0, gc0, _gr1, _gc1 = container["grey_bbox"]
        yellow = container["yellow"]
        grey_mask = container["grey_mask"]

        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = BG

        if color == RED:
            axis, _ = _direction_from_marker(yellow, container["grey_bbox"])
            if axis in ("left", "right"):
                transformed = np.fliplr(grey_mask)
            else:
                transformed = np.flipud(grey_mask)
            _paint_mask(out, gr0, gc0, transformed, GREY)

        elif color == BLUE:
            transformed, top, left = _rotate_cw_about_center(grey_mask, gr0, gc0)
            _paint_mask(out, top, left, transformed, GREY)

        else:
            _name, (dr, dc) = _direction_from_marker(yellow, container["grey_bbox"])
            moved = list(container["grey_cells"])
            while not _adjacent_to_marker(moved, yellow):
                candidate = [(r + dr, c + dc) for r, c in moved]
                assert all(fr0 < r < fr1 and fc0 < c < fc1 for r, c in candidate)
                assert yellow not in candidate
                moved = candidate
            out[yellow] = YELLOW
            for r, c in moved:
                out[r, c] = GREY

    return out


def _red_lr_shape() -> np.ndarray:
    return _sample_shape(
        lambda m: m.shape[1] > 1 and not np.array_equal(m, np.fliplr(m)),
        [(2, 2), (2, 3), (3, 2), (3, 3)],
    )


def _red_ud_shape() -> np.ndarray:
    return _sample_shape(
        lambda m: m.shape[0] > 1 and not np.array_equal(m, np.flipud(m)),
        [(2, 2), (2, 3), (3, 2), (3, 3)],
    )


def _blue_shape() -> np.ndarray:
    return _sample_shape(
        lambda m: m.shape[0] == m.shape[1] and m.shape[0] in (2, 3) and not np.array_equal(m, np.rot90(m, -1)),
        [(2, 2), (3, 3)],
    )


def _green_shape() -> np.ndarray:
    return _sample_shape(lambda _m: True, [(2, 2), (2, 3), (3, 2), (3, 3)])


def _empty_container(interior_h: int, interior_w: int, border_color: int) -> np.ndarray:
    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    return patch


def _make_red_container(axis: str) -> np.ndarray:
    for _ in range(300):
        interior_h = random.randint(5, 8)
        interior_w = random.randint(5, 8)
        shape = _red_lr_shape() if axis == "lr" else _red_ud_shape()
        sh, sw = shape.shape
        if sh > interior_h or sw > interior_w:
            continue

        if axis == "lr":
            side = random.choice(["left", "right"])
            sr = random.randint(0, interior_h - sh)
            if side == "left":
                if interior_w - sw < 1:
                    continue
                sc = random.randint(1, interior_w - sw)
                gap = random.randint(1, sc)
                yellow = (sr + random.randrange(sh), sc - gap)
            else:
                max_sc = interior_w - sw - 1
                if max_sc < 0:
                    continue
                sc = random.randint(0, max_sc)
                gap = random.randint(1, interior_w - (sc + sw))
                yellow = (sr + random.randrange(sh), sc + sw - 1 + gap)
        else:
            side = random.choice(["up", "down"])
            sc = random.randint(0, interior_w - sw)
            if side == "up":
                if interior_h - sh < 1:
                    continue
                sr = random.randint(1, interior_h - sh)
                gap = random.randint(1, sr)
                yellow = (sr - gap, sc + random.randrange(sw))
            else:
                max_sr = interior_h - sh - 1
                if max_sr < 0:
                    continue
                sr = random.randint(0, max_sr)
                gap = random.randint(1, interior_h - (sr + sh))
                yellow = (sr + sh - 1 + gap, sc + random.randrange(sw))

        patch = _empty_container(interior_h, interior_w, RED)
        _paint_mask(patch, 1 + sr, 1 + sc, shape, GREY)
        patch[1 + yellow[0], 1 + yellow[1]] = YELLOW
        return patch
    raise RuntimeError("failed to build red container")


def _make_blue_container() -> np.ndarray:
    for _ in range(300):
        interior_h = random.randint(5, 8)
        interior_w = random.randint(5, 8)
        shape = _blue_shape()
        sh, sw = shape.shape
        if sh > interior_h or sw > interior_w:
            continue
        sr = random.randint(0, interior_h - sh)
        sc = random.randint(0, interior_w - sw)
        empties = [
            (r, c)
            for r in range(interior_h)
            for c in range(interior_w)
            if not (sr <= r < sr + sh and sc <= c < sc + sw and shape[r - sr, c - sc])
        ]
        if not empties:
            continue
        yellow = random.choice(empties)
        patch = _empty_container(interior_h, interior_w, BLUE)
        _paint_mask(patch, 1 + sr, 1 + sc, shape, GREY)
        patch[1 + yellow[0], 1 + yellow[1]] = YELLOW
        return patch
    raise RuntimeError("failed to build blue container")


def _make_green_container(direction: str) -> np.ndarray:
    for _ in range(300):
        interior_h = random.randint(5, 8)
        interior_w = random.randint(5, 8)
        shape = _green_shape()
        sh, sw = shape.shape
        if sh > interior_h or sw > interior_w:
            continue

        if direction == "left":
            side_rows = [r for r in range(sh) if shape[r, 0]]
            if not side_rows or interior_w - sw < 2:
                continue
            sr = random.randint(0, interior_h - sh)
            sc = random.randint(2, interior_w - sw)
            gap = random.randint(2, sc)
            yellow = (sr + random.choice(side_rows), sc - gap)
        elif direction == "right":
            side_rows = [r for r in range(sh) if shape[r, sw - 1]]
            max_sc = interior_w - sw - 2
            if not side_rows or max_sc < 0:
                continue
            sr = random.randint(0, interior_h - sh)
            sc = random.randint(0, max_sc)
            gap = random.randint(2, interior_w - (sc + sw))
            yellow = (sr + random.choice(side_rows), sc + sw - 1 + gap)
        elif direction == "up":
            side_cols = [c for c in range(sw) if shape[0, c]]
            if not side_cols or interior_h - sh < 2:
                continue
            sr = random.randint(2, interior_h - sh)
            sc = random.randint(0, interior_w - sw)
            gap = random.randint(2, sr)
            yellow = (sr - gap, sc + random.choice(side_cols))
        else:
            side_cols = [c for c in range(sw) if shape[sh - 1, c]]
            max_sr = interior_h - sh - 2
            if not side_cols or max_sr < 0:
                continue
            sr = random.randint(0, max_sr)
            sc = random.randint(0, interior_w - sw)
            gap = random.randint(2, interior_h - (sr + sh))
            yellow = (sr + sh - 1 + gap, sc + random.choice(side_cols))

        patch = _empty_container(interior_h, interior_w, GREEN)
        _paint_mask(patch, 1 + sr, 1 + sc, shape, GREY)
        patch[1 + yellow[0], 1 + yellow[1]] = YELLOW
        return patch
    raise RuntimeError("failed to build green container")


def _make_container(spec: tuple[str, str | None]) -> np.ndarray:
    kind, variant = spec
    if kind == "red":
        assert variant in {"lr", "ud"}
        return _make_red_container(variant)
    if kind == "blue":
        return _make_blue_container()
    assert kind == "green"
    assert variant in {"left", "right", "up", "down"}
    return _make_green_container(variant)


def _can_place_rect(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    if top < 1 or left < 1 or top + h >= hh or left + w >= ww:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(hh, top + h + 1)
    c1 = min(ww, left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_patches(shape: tuple[int, int], patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        ph, pw = patches[idx].shape
        candidates = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
            if _can_place_rect(occupied, r, c, ph, pw)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[idx] = (top, left)
        occupied[top : top + ph, left : left + pw] = True

    return [pos for pos in placements if pos is not None]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _gen_attempt() -> np.ndarray | None:
    specs: list[tuple[str, str | None]] = [
        ("red", "lr"),
        ("red", "ud"),
        ("blue", None),
        ("green", random.choice(["left", "right", "up", "down"])),
    ]
    for _ in range(random.choice([0, 1])):
        extra = random.choice(
            [
                ("red", random.choice(["lr", "ud"])),
                ("blue", None),
                ("green", random.choice(["left", "right", "up", "down"])),
            ]
        )
        specs.append(extra)
    random.shuffle(specs)

    try:
        patches = [_make_container(spec) for spec in specs]
    except RuntimeError:
        return None

    grid_h = random.randint(20, 30)
    grid_w = random.randint(20, 30)
    placements = _place_patches((grid_h, grid_w), patches)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for patch, (top, left) in zip(patches, placements):
        _paste_patch(grid, patch, top, left)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
