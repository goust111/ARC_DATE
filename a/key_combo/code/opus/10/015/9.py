from __future__ import annotations

from collections import deque
import random

import numpy as np


TEMPLATES = (
    (
        6,
        (
            ".#.",
            "###",
            "#.#",
        ),
    ),
    (
        7,
        (
            "##.",
            ".##",
            "###",
        ),
    ),
    (
        8,
        (
            "###",
            "#.#",
            "###",
        ),
    ),
    (
        9,
        (
            ".##",
            "###",
            "###",
            ".#.",
        ),
    ),
    (
        10,
        (
            ".###",
            "####",
            "###.",
        ),
    ),
    (
        11,
        (
            "####",
            "#.##",
            "####",
        ),
    ),
)

BACKGROUND = 0
GEN_ATTEMPTS = 160
SETTLED_ATTEMPTS = 48


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    return coords - coords.min(axis=0, keepdims=True)


def _pattern_to_coords(pattern: tuple[str, ...]) -> np.ndarray:
    coords = []
    for r, row in enumerate(pattern):
        for c, ch in enumerate(row):
            if ch == "#":
                coords.append((r, c))
    arr = np.asarray(coords, dtype=int)
    return _normalize(arr)


def _random_transform(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    if random.random() < 0.5:
        arr = arr[:, [1, 0]]
    if random.random() < 0.5:
        arr[:, 0] *= -1
    if random.random() < 0.5:
        arr[:, 1] *= -1
    return _normalize(arr)


def _shift(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.asarray(coords, dtype=int) + np.array([int(dr), int(dc)], dtype=int)


def _layout_signature(objects: list[dict]) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    sig = []
    for obj in objects:
        coords = tuple(sorted((int(r), int(c)) for r, c in obj["coords"]))
        sig.append((int(obj["color"]), coords))
    return tuple(sorted(sig))


def _paint(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return out


def _occupancy(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = np.asarray(obj["coords"], dtype=int)
        if np.any(occ[coords[:, 0], coords[:, 1]] != -1):
            raise AssertionError("objects must not overlap")
        occ[coords[:, 0], coords[:, 1]] = idx
    return occ


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            colors = {int(grid[rr, cc]) for rr, cc in arr}
            if len(colors) != 1:
                raise AssertionError("each sprite must be monochromatic")
            if next(iter(colors)) != color:
                raise AssertionError("component color mismatch")
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                }
            )
    return objects


def _movable_down(objects: list[dict], occ: np.ndarray) -> set[int]:
    h, _w = occ.shape
    movable = set(range(len(objects)))
    changed = True
    while changed:
        changed = False
        for idx in list(movable):
            coords = np.asarray(objects[idx]["coords"], dtype=int)
            shifted_rows = coords[:, 0] + 1
            if np.any(shifted_rows >= h):
                movable.remove(idx)
                changed = True
                continue
            labels = occ[shifted_rows, coords[:, 1]]
            blockers = {int(label) for label in labels if int(label) not in {-1, idx}}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(idx)
                changed = True
    return movable


def _settle_objects(objects: list[dict], shape: tuple[int, int]) -> list[dict]:
    settled = [
        {
            "color": int(obj["color"]),
            "coords": np.asarray(obj["coords"], dtype=int).copy(),
            "area": int(obj["area"]),
        }
        for obj in objects
    ]
    h, _w = shape
    for _ in range(h + 1):
        occ = _occupancy(shape, settled)
        movable = _movable_down(settled, occ)
        if not movable:
            return settled
        for idx in movable:
            settled[idx]["coords"][:, 0] += 1
    raise AssertionError("gravity did not settle")


def _bottom_row(coords: np.ndarray) -> int:
    return int(np.max(coords[:, 0]))


def _rings(colors: list[int]) -> np.ndarray:
    n = len(colors)
    side = 2 * n - 1
    out = np.full((side, side), BACKGROUND, dtype=int)
    for layer, color in enumerate(colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the gravity-settled foreground sprites into nested square rings ordered by landing height.

    Input structure:
    - The grid has a black background and 4-5 separate monochromatic 4-connected sprites.
    - Sprites have distinct non-zero colors, irregular non-rectangular shapes, and some overlapping column spans.
    - The sprites are scattered at different heights and act as rigid bodies under downward gravity.

    Rule:
    - Simultaneously apply one-row downward gravity steps: a sprite moves only when a one-row shift stays in bounds
      and is not blocked by the bottom border or by any sprite that has already become unable to move.
    - Repeat until every sprite has settled, then sort the settled sprites by their bottom row from largest to smallest.
    - Create a (2N-1) x (2N-1) output for N sprites, filling the outer ring with the lowest settled sprite color,
      the next ring with the next settled color, and so on until the highest settled sprite becomes the center pixel.

    Key insight:
    - The output order is determined by the full gravity-and-collision simulation, not by sprite area, original height,
      or original shape; only the settled vertical stacking order and colors survive the compression.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("expected a non-empty 2D grid")

    objects = _components(g, BACKGROUND)
    if not (4 <= len(objects) <= 5):
        raise AssertionError("expected 4 or 5 sprites")

    settled = _settle_objects(objects, g.shape)
    bottoms = [_bottom_row(obj["coords"]) for obj in settled]
    if len(set(bottoms)) != len(bottoms):
        raise AssertionError("settled bottom rows must be unique")

    ordered = sorted(settled, key=lambda obj: _bottom_row(obj["coords"]), reverse=True)
    return _rings([int(obj["color"]) for obj in ordered])


def _sample_objects(n: int) -> list[dict]:
    chosen = random.sample(TEMPLATES, n)
    colors = random.sample(list(range(1, 10)), n)
    objects = []
    for color, (area, pattern) in zip(colors, chosen):
        local = _random_transform(_pattern_to_coords(pattern))
        r0, c0, r1, c1 = _bbox(local)
        if r1 == 0 or c1 == 0:
            raise AssertionError("sprite must span at least 2 rows and 2 cols")
        if local.shape[0] != area:
            raise AssertionError("template area mismatch")
        objects.append({"color": int(color), "local": local, "area": int(area)})
    return objects


def _drop_row(local: np.ndarray, col: int, occ: np.ndarray) -> int | None:
    h, w = occ.shape
    _r0, _c0, r1, c1 = _bbox(local)
    height = r1 + 1
    width = c1 + 1
    if col < 0 or col + width > w:
        return None
    row = 0
    placed = _shift(local, row, col)
    if np.any(occ[placed[:, 0], placed[:, 1]] != -1):
        return None
    while row + height < h:
        nxt = _shift(local, row + 1, col)
        if np.any(occ[nxt[:, 0], nxt[:, 1]] != -1):
            break
        row += 1
    return int(row)


def _support_info(coords: np.ndarray, occ: np.ndarray, h: int) -> tuple[bool, set[int]]:
    floor = False
    supporters: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr + 1 >= h:
            floor = True
        else:
            below = int(occ[rr + 1, cc])
            if below >= 0:
                supporters.add(below)
    return floor, supporters


def _column_overlap(a: np.ndarray, b: np.ndarray) -> int:
    _ar0, ac0, _ar1, ac1 = _bbox(a)
    _br0, bc0, _br1, bc1 = _bbox(b)
    return max(0, min(ac1, bc1) - max(ac0, bc0) + 1)


def _has_column_overlap(objects: list[dict]) -> bool:
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _column_overlap(objects[i]["coords"], objects[j]["coords"]) > 0:
                return True
    return False


def _build_settled_layout(base_objects: list[dict], shape: tuple[int, int]) -> tuple[list[int], list[dict]] | None:
    h, w = shape
    area_desc = [idx for idx, _obj in sorted(enumerate(base_objects), key=lambda x: x[1]["area"], reverse=True)]

    for _ in range(SETTLED_ATTEMPTS):
        final_order = random.sample(list(range(len(base_objects))), len(base_objects))
        if final_order == area_desc:
            continue

        placed: list[dict] = []
        occ = np.full((h, w), -1, dtype=int)
        previous_bottom = h

        for rank, obj_idx in enumerate(final_order):
            local = np.asarray(base_objects[obj_idx]["local"], dtype=int)
            candidates = []
            _r0, _c0, _r1, c1 = _bbox(local)
            width = c1 + 1
            for col in range(0, w - width + 1):
                row = _drop_row(local, col, occ)
                if row is None:
                    continue
                coords = _shift(local, row, col)
                bottom = _bottom_row(coords)
                if bottom >= previous_bottom:
                    continue
                floor, supporters = _support_info(coords, occ, h)
                if rank == 0:
                    if not floor:
                        continue
                    center_bias = abs((2 * col + width - 1) - (w - 1))
                    candidates.append((bottom, -center_bias, coords))
                    continue
                if floor or (rank - 1) not in supporters:
                    continue
                overlap = _column_overlap(coords, placed[-1]["coords"])
                if overlap <= 0:
                    continue
                _pr0, pc0, _pr1, pc1 = _bbox(placed[-1]["coords"])
                center_bias = abs((2 * col + width - 1) - (pc0 + pc1))
                candidates.append((bottom, overlap, -center_bias, coords))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[:-1], reverse=True)
            coords = np.asarray(random.choice(candidates[: min(4, len(candidates))])[-1], dtype=int)
            placed.append(
                {
                    "color": int(base_objects[obj_idx]["color"]),
                    "coords": coords,
                    "area": int(base_objects[obj_idx]["area"]),
                    "source_idx": int(obj_idx),
                }
            )
            occ[coords[:, 0], coords[:, 1]] = len(placed) - 1
            previous_bottom = _bottom_row(coords)

        if len(placed) != len(base_objects):
            continue

        bottoms = [_bottom_row(obj["coords"]) for obj in placed]
        if len(set(bottoms)) != len(bottoms):
            continue

        top_margin = min(int(np.min(obj["coords"][:, 0])) for obj in placed)
        if top_margin < len(base_objects) - 1:
            continue

        return final_order, placed

    return None


def _separated(a: np.ndarray, b: np.ndarray) -> bool:
    cells_b = {(int(r), int(c)) for r, c in b}
    for r, c in a:
        rr = int(r)
        cc = int(c)
        if (rr, cc) in cells_b:
            return False
        for nr, nc in _neighbors4(rr, cc):
            if (nr, nc) in cells_b:
                return False
    return True


def _supports_collision(settled: list[dict], shape: tuple[int, int]) -> bool:
    occ = _occupancy(shape, settled)
    h, _w = shape
    for idx, obj in enumerate(settled):
        coords = np.asarray(obj["coords"], dtype=int)
        mask_occ = np.where(occ == idx, -1, occ)
        floor, supporters = _support_info(coords, mask_occ, h)
        if not floor and supporters:
            return True
    return False


def _lift_input_layout(settled: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    ordered = sorted(
        enumerate(settled),
        key=lambda item: _bottom_row(item[1]["coords"]),
        reverse=True,
    )
    slack = min(int(np.min(obj["coords"][:, 0])) - rank for rank, (_idx, obj) in enumerate(ordered))
    if slack < 0:
        return None
    base_lift = random.randint(0, slack)

    lifted: list[dict] = []
    for rank, (_idx, obj) in enumerate(ordered):
        coords = np.asarray(obj["coords"], dtype=int)
        lift = base_lift + rank
        lifted.append(
            {
                "color": int(obj["color"]),
                "coords": _shift(coords, -lift, 0),
                "area": int(obj["area"]),
                "source_idx": int(obj["source_idx"]),
            }
        )

    for i in range(len(lifted)):
        for j in range(i + 1, len(lifted)):
            if not _separated(lifted[i]["coords"], lifted[j]["coords"]):
                return None

    settled_again = _settle_objects(lifted, shape)
    if _layout_signature(settled_again) != _layout_signature(settled):
        return None
    if not _supports_collision(settled_again, shape):
        return None
    return lifted


def _validate_case(inp: np.ndarray, settled: list[dict], base_objects: list[dict], final_order: list[int]) -> bool:
    objects = _components(inp, BACKGROUND)
    if len(objects) != len(base_objects):
        return False
    if len({obj["color"] for obj in objects}) != len(objects):
        return False
    if any(obj["area"] < 6 or obj["area"] > 15 for obj in objects):
        return False
    if any((_bbox(obj["coords"])[2] - _bbox(obj["coords"])[0] == 0) or (_bbox(obj["coords"])[3] - _bbox(obj["coords"])[1] == 0) for obj in objects):
        return False
    if any(obj["area"] == (1 + _bbox(obj["coords"])[2] - _bbox(obj["coords"])[0]) * (1 + _bbox(obj["coords"])[3] - _bbox(obj["coords"])[1]) for obj in objects):
        return False
    if not _has_column_overlap(objects):
        return False

    settled_main = _settle_objects(objects, inp.shape)
    if _layout_signature(settled_main) != _layout_signature(settled):
        return False

    bottoms = [_bottom_row(obj["coords"]) for obj in settled_main]
    if len(set(bottoms)) != len(bottoms):
        return False

    if not _supports_collision(settled_main, inp.shape):
        return False

    area_desc = [idx for idx, _obj in sorted(enumerate(base_objects), key=lambda x: x[1]["area"], reverse=True)]
    if list(final_order) == area_desc:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(15, 18)
        w = random.randint(13, 16)
        n = random.randint(4, 5)

        base_objects = _sample_objects(n)
        built = _build_settled_layout(base_objects, (h, w))
        if built is None:
            continue
        final_order, settled = built

        lifted = _lift_input_layout(settled, (h, w))
        if lifted is None:
            continue

        inp = _paint((h, w), lifted)
        if _validate_case(inp, settled, base_objects, final_order):
            return inp

    raise RuntimeError("failed to generate a non-degenerate gravity-settle input")
