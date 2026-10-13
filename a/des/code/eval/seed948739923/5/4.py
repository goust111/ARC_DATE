from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
CONTAINER = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GAP_SIDES = ("top", "right", "bottom", "left")
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components from a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _patch_from_coords(coords: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch


def _parse_container(grid: np.ndarray, coords: np.ndarray) -> dict:
    cells = {tuple(map(int, xy)) for xy in coords}
    r0, c0, r1, c1 = _bbox(coords)

    perimeter = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    missing = sorted(perimeter - cells)
    assert cells <= perimeter
    assert len(missing) == 2

    gap_side: str | None = None
    if missing[0][0] == missing[1][0] and abs(missing[0][1] - missing[1][1]) == 1:
        if missing[0][0] == r0:
            gap_side = "top"
        elif missing[0][0] == r1:
            gap_side = "bottom"
    if gap_side is None and missing[0][1] == missing[1][1] and abs(missing[0][0] - missing[1][0]) == 1:
        if missing[0][1] == c0:
            gap_side = "left"
        elif missing[0][1] == c1:
            gap_side = "right"
    assert gap_side is not None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mark_coords = np.argwhere((inner != BG) & (inner != CONTAINER))
    assert mark_coords.shape == (1, 2)
    mr = int(mark_coords[0, 0] + r0 + 1)
    mc = int(mark_coords[0, 1] + c0 + 1)

    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "gap_side": gap_side,
        "marker_pos": (mr, mc),
        "color": int(grid[mr, mc]),
    }


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers = [_parse_container(grid, comp) for comp in _components(grid == CONTAINER)]
    containers.sort(key=lambda d: (d["r0"], d["c0"]))
    assert 2 <= len(containers) <= 4
    return containers


def _find_external_sprites(grid: np.ndarray, containers: list[dict]) -> list[dict]:
    marker_cells = {info["marker_pos"] for info in containers}
    sprites: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {BG, CONTAINER}):
        for comp in _components(grid == int(color)):
            cells = {tuple(map(int, xy)) for xy in comp}
            if cells == {next(iter(cells))} and next(iter(cells)) in marker_cells:
                continue
            assert cells.isdisjoint(marker_cells)
            sprites.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "patch": _patch_from_coords(comp, int(color)),
                }
            )
    assert sprites
    return sprites


def _is_passable(patch: np.ndarray) -> bool:
    return min(map(int, patch.shape)) <= 2


def _orient_for_gap(patch: np.ndarray, gap_side: str) -> np.ndarray:
    h, w = map(int, patch.shape)
    if gap_side in {"top", "bottom"}:
        if w <= 2:
            return patch.copy()
        assert h <= 2
        return np.rot90(patch, 1)
    if h <= 2:
        return patch.copy()
    assert w <= 2
    return np.rot90(patch, 1)


def _placement(container: dict, patch: np.ndarray) -> tuple[int, int]:
    ph, pw = map(int, patch.shape)
    inner_h = int(container["h"] - 2)
    inner_w = int(container["w"] - 2)
    assert ph <= inner_h and pw <= inner_w

    if container["gap_side"] == "top":
        top = int(container["r1"] - ph)
        left = int(container["c0"] + 1 + (inner_w - pw) // 2)
    elif container["gap_side"] == "bottom":
        top = int(container["r0"] + 1)
        left = int(container["c0"] + 1 + (inner_w - pw) // 2)
    elif container["gap_side"] == "left":
        top = int(container["r0"] + 1 + (inner_h - ph) // 2)
        left = int(container["c1"] - pw)
    else:
        top = int(container["r0"] + 1 + (inner_h - ph) // 2)
        left = int(container["c0"] + 1)
    return top, left


def _apply_liquid(out: np.ndarray, container: dict, sprite: dict) -> None:
    for r, c in sprite["coords"]:
        out[int(r), int(c)] = BG

    patch = _orient_for_gap(sprite["patch"], str(container["gap_side"]))
    top, left = _placement(container, patch)
    ph, pw = patch.shape
    target = out[top : top + ph, left : left + pw]
    mask = patch != BG
    assert target.shape == patch.shape
    assert not bool(np.any((target == CONTAINER) & mask))
    target[mask] = int(sprite["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each container's marker color to an outside sprite, pick the unique
    same-color sprite that can fit through the 2-cell gap (with optional 90°
    rotation), then move it inside against the wall opposite the gap.
    """
    g = np.asarray(grid, dtype=int)
    containers = _find_containers(g)
    sprites = _find_external_sprites(g, containers)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        by_color[int(sprite["color"])].append(sprite)

    out = g.copy()
    for container in containers:
        candidates = by_color[int(container["color"])]
        passable = [sprite for sprite in candidates if _is_passable(sprite["patch"])]
        assert len(passable) == 1
        _apply_liquid(out, container, passable[0])
    return out


def _container_patch(h: int, w: int, gap_side: str, gap_start: int, marker_color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = CONTAINER
    patch[-1, :] = CONTAINER
    patch[:, 0] = CONTAINER
    patch[:, -1] = CONTAINER

    if gap_side == "top":
        patch[0, gap_start : gap_start + 2] = BG
    elif gap_side == "bottom":
        patch[-1, gap_start : gap_start + 2] = BG
    elif gap_side == "left":
        patch[gap_start : gap_start + 2, 0] = BG
    else:
        patch[gap_start : gap_start + 2, -1] = BG

    patch[h // 2, w // 2] = int(marker_color)
    return patch


def _rotate_random(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


_BLOCKING_MASKS = [
    np.ones((3, 3), dtype=bool),
    np.ones((3, 4), dtype=bool),
    np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    ),
]


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _selected_patch(gap_side: str, needs_rotation: bool, color: int) -> np.ndarray:
    if gap_side in {"top", "bottom"}:
        shape = (2, 3) if needs_rotation else (3, 2)
    else:
        shape = (3, 2) if needs_rotation else (2, 3)
    return np.full(shape, int(color), dtype=int)


def _blocking_patch(color: int) -> np.ndarray:
    mask = _rotate_random(random.choice(_BLOCKING_MASKS))
    return _paint_mask(mask, color)


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _occupy(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    target = grid[r0 : r0 + h, c0 : c0 + w]
    assert target.shape == patch.shape
    assert not bool(np.any((target != BG) & mask))
    target[mask] = patch[mask]


def _choose_gap_start(side_len: int) -> int:
    choices = list(range(1, side_len - 2))
    assert choices
    return random.choice(choices)


def _expected_from_spec(inp: np.ndarray, containers: list[dict], sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    chosen_by_color = {int(sprite["color"]): sprite for sprite in sprites if sprite["selected"]}
    for container in containers:
        _apply_liquid(out, container, chosen_by_color[int(container["color"])])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    colors = random.sample(MARKER_COLORS, n_containers)

    min_side = 14 + 4 * n_containers
    H = random.randint(min_side, 30)
    W = random.randint(min_side, 30)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    containers: list[dict] = []
    sprites: list[dict] = []

    for color in colors:
        h = random.choice((5, 7))
        w = random.choice((5, 7))
        gap_side = random.choice(GAP_SIDES)
        gap_len = w if gap_side in {"top", "bottom"} else h
        gap_start = _choose_gap_start(gap_len)
        patch = _container_patch(h, w, gap_side, gap_start, int(color))

        placed = False
        for _ in range(200):
            r0 = random.randint(0, H - h)
            c0 = random.randint(0, W - w)
            if not _can_place(occupied, r0, c0, h, w, pad=1):
                continue
            _place_patch(grid, patch, r0, c0)
            _occupy(occupied, r0, c0, h, w)
            containers.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + h - 1,
                    "c1": c0 + w - 1,
                    "h": h,
                    "w": w,
                    "gap_side": gap_side,
                    "marker_pos": (r0 + h // 2, c0 + w // 2),
                    "color": int(color),
                }
            )
            placed = True
            break
        if not placed:
            return None

        needs_rotation = random.random() < 0.5
        selected = _selected_patch(gap_side, needs_rotation, int(color))
        sprite_choices = [selected]
        for _ in range(random.randint(1, 2)):
            sprite_choices.append(_blocking_patch(int(color)))

        random.shuffle(sprite_choices)
        selected_index = next(i for i, patch_i in enumerate(sprite_choices) if np.array_equal(patch_i, selected))
        for idx, patch_i in enumerate(sprite_choices):
            sh, sw = patch_i.shape
            placed_sprite = False
            for _ in range(200):
                sr = random.randint(0, H - sh)
                sc = random.randint(0, W - sw)
                if not _can_place(occupied, sr, sc, sh, sw, pad=1):
                    continue
                _place_patch(grid, patch_i, sr, sc)
                _occupy(occupied, sr, sc, sh, sw)
                coords = np.argwhere(patch_i != BG) + np.array([[sr, sc]])
                sprites.append(
                    {
                        "color": int(color),
                        "coords": coords.astype(int),
                        "patch": patch_i.copy(),
                        "selected": idx == selected_index,
                    }
                )
                placed_sprite = True
                break
            if not placed_sprite:
                return None

    expected = _expected_from_spec(grid, containers, sprites)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance")
