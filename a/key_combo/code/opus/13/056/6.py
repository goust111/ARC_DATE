from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ["right", "left", "down", "up"]
DIR_STEP = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
DIR_TO_CROP_SIDE = {
    "right": "left",
    "left": "right",
    "down": "top",
    "up": "bottom",
}
OTHER_AXIS_SIDE = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}


def _neighbors(r: int, c: int, connectivity: int = 4) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int, *, mask: np.ndarray | None = None) -> None:
    h, w = src.shape
    if mask is None:
        dst[r0 : r0 + h, c0 : c0 + w] = src
    else:
        view = dst[r0 : r0 + h, c0 : c0 + w]
        view[mask] = src[mask]


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _monochrome_components(grid: np.ndarray, colors: Iterable[int], *, connectivity: int = 4) -> list[dict]:
    out: list[dict] = []
    for color in colors:
        for coords in _components_mask(grid == int(color), connectivity=connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _vector_direction(dr: int, dc: int) -> str | None:
    if dr == 0 and dc > 0:
        return "right"
    if dr == 0 and dc < 0:
        return "left"
    if dc == 0 and dr > 0:
        return "down"
    if dc == 0 and dr < 0:
        return "up"
    return None


def _render_local_coords(coords: np.ndarray, shape: tuple[int, int], color: int) -> np.ndarray:
    arr = np.zeros(shape, dtype=int)
    if coords.size:
        arr[coords[:, 0], coords[:, 1]] = int(color)
    return arr


def _crop_local_coords(
    coords: np.ndarray,
    full_shape: tuple[int, int],
    crop_side: str,
    amount: int,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    h, w = map(int, full_shape)
    amount = int(amount)
    if crop_side == "left":
        if amount >= w:
            return None
        kept = coords[coords[:, 1] >= amount].copy()
        kept[:, 1] -= amount
        return kept, (h, w - amount)
    if crop_side == "right":
        if amount >= w:
            return None
        kept = coords[coords[:, 1] < w - amount].copy()
        return kept, (h, w - amount)
    if crop_side == "top":
        if amount >= h:
            return None
        kept = coords[coords[:, 0] >= amount].copy()
        kept[:, 0] -= amount
        return kept, (h - amount, w)
    if crop_side == "bottom":
        if amount >= h:
            return None
        kept = coords[coords[:, 0] < h - amount].copy()
        return kept, (h - amount, w)
    raise ValueError(f"unknown crop_side {crop_side}")


def _spans_shape(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    if coords.size == 0:
        return False
    return _bbox_coords(coords) == (0, 0, int(shape[0]) - 1, int(shape[1]) - 1)


def _best_cover_translation(full_coords: np.ndarray, part_coords: np.ndarray) -> tuple[int, int]:
    full = {tuple(map(int, rc)) for rc in full_coords}
    part = {tuple(map(int, rc)) for rc in part_coords}
    valid: set[tuple[int, int]] = set()
    for pr, pc in part:
        for sr, sc in full:
            dr = int(pr - sr)
            dc = int(pc - sc)
            if dr == 0 and dc == 0:
                continue
            if (dr == 0) == (dc == 0):
                continue
            translated = {(r + dr, c + dc) for r, c in full}
            if part.issubset(translated):
                valid.add((dr, dc))
    assert valid, "expected at least one valid cover translation"
    best_norm = max(dr * dr + dc * dc for dr, dc in valid)
    bests = sorted((dr, dc) for dr, dc in valid if dr * dr + dc * dc == best_norm)
    assert len(bests) == 1, "expected a unique best cover translation"
    return bests[0]


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components_mask(grid == FRAME_COLOR, connectivity=4)
    assert comps, "expected a frame component"
    largest = max(comps, key=lambda comp: comp.shape[0])
    return _bbox_coords(largest)


def _rectangular_cavities(frame_crop: np.ndarray) -> list[dict]:
    holes: list[dict] = []
    for coords in _components_mask(frame_crop == BACKGROUND, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r0 == 0 or c0 == 0 or r1 == frame_crop.shape[0] - 1 or c1 == frame_crop.shape[1] - 1:
            continue
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if coords.shape[0] != h * w:
            continue
        holes.append({"bbox": (r0, c0, r1, c1), "shape": (h, w), "coords": coords})
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each offset partial copy into a crop direction, trim the matching sprite, and inlay it into the unique cavity.

    Input structure:
    - A black background surrounds one large grey rectangular frame with 2-3 black rectangular cavities inside it.
    - Outside the frame, each non-grey color appears as one larger 4-connected sprite plus one smaller same-color partial copy.
    - Every partial copy is a translated connected subset of its sprite, and the translation is purely up, down, left, or right.

    Rule:
    - Crop the output to the grey frame bbox and measure every internal rectangular cavity.
    - For each color, align the larger sprite against the smaller copy and take the unique farthest valid cover translation as the displacement direction.
    - Use that direction to choose the crop side on the opposite edge of the sprite, crop by the unique amount that makes its bbox match exactly one cavity, then paste the cropped sprite into that cavity with top-left alignment.

    Key insight:
    - The partial copy does not say where to repeat the sprite; it only reveals which side must be trimmed off before the sprite can fit a cavity of the right size.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox = _frame_bbox(g)
    r0, c0, r1, c1 = frame_bbox
    frame_crop = _crop_bbox(g, frame_bbox)
    cavities = _rectangular_cavities(frame_crop)
    out = frame_crop.copy()

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND
    colors = sorted(int(x) for x in np.unique(outside) if x not in (BACKGROUND, FRAME_COLOR))
    used_cavities: set[int] = set()

    for color in colors:
        comps = _components_mask(outside == color, connectivity=4)
        if len(comps) != 2:
            continue
        comps = sorted(comps, key=lambda comp: comp.shape[0], reverse=True)
        full_abs = comps[0]
        part_abs = comps[1]
        dr, dc = _best_cover_translation(full_abs, part_abs)
        direction = _vector_direction(dr, dc)
        assert direction is not None, "translation must be cardinal"
        crop_side = DIR_TO_CROP_SIDE[direction]

        fr0, fc0, fr1, fc1 = _bbox_coords(full_abs)
        full_shape = (int(fr1 - fr0 + 1), int(fc1 - fc0 + 1))
        full_local = full_abs - np.array([fr0, fc0], dtype=int)

        matches: list[tuple[int, np.ndarray]] = []
        for idx, cavity in enumerate(cavities):
            if idx in used_cavities:
                continue
            ch, cw = map(int, cavity["shape"])
            if crop_side in ("left", "right"):
                amount = full_shape[1] - cw
                if ch != full_shape[0] or amount not in (1, 2):
                    continue
            else:
                amount = full_shape[0] - ch
                if cw != full_shape[1] or amount not in (1, 2):
                    continue
            cropped = _crop_local_coords(full_local, full_shape, crop_side, amount)
            if cropped is None:
                continue
            cropped_coords, cropped_shape = cropped
            if tuple(map(int, cropped_shape)) != (ch, cw):
                continue
            if not _spans_shape(cropped_coords, (ch, cw)):
                continue
            patch = _render_local_coords(cropped_coords, (ch, cw), color)
            matches.append((idx, patch))

        if len(matches) != 1:
            continue
        cavity_idx, patch = matches[0]
        used_cavities.add(cavity_idx)
        cr0, cc0, _cr1, _cc1 = cavities[cavity_idx]["bbox"]
        _paste(out, patch, cr0, cc0, mask=patch != BACKGROUND)

    return out


def _random_connected_shape(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    while len(cells) < area:
        frontier.clear()
        for r, c in cells:
            for nr, nc in _neighbors(r, c, 4):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return np.empty((0, 2), dtype=int)
        cells.add(random.choice(tuple(frontier)))
    return np.array(sorted(cells), dtype=int)


def _edge_subset(coords: np.ndarray, full_shape: tuple[int, int], direction: str) -> np.ndarray | None:
    h, w = map(int, full_shape)
    if direction == "right":
        edge = coords[coords[:, 1] == w - 1]
        if edge.shape[0] == 2 and abs(int(edge[0, 0]) - int(edge[1, 0])) == 1:
            return edge
    elif direction == "left":
        edge = coords[coords[:, 1] == 0]
        if edge.shape[0] == 2 and abs(int(edge[0, 0]) - int(edge[1, 0])) == 1:
            return edge
    elif direction == "down":
        edge = coords[coords[:, 0] == h - 1]
        if edge.shape[0] == 2 and abs(int(edge[0, 1]) - int(edge[1, 1])) == 1:
            return edge
    elif direction == "up":
        edge = coords[coords[:, 0] == 0]
        if edge.shape[0] == 2 and abs(int(edge[0, 1]) - int(edge[1, 1])) == 1:
            return edge
    return None


def _generate_sprite_spec(direction: str, color: int, used_shapes: set[tuple[int, int]]) -> dict | None:
    horizontal = direction in ("right", "left")
    crop_side = DIR_TO_CROP_SIDE[direction]
    step = DIR_STEP[direction]

    for _ in range(400):
        amount = random.choice([1, 2])
        if horizontal:
            keep_h = random.randint(2, 3)
            keep_w = random.randint(2, 4 - (1 if amount == 2 else 0))
            full_shape = (keep_h, keep_w + amount)
            keep_shape = (keep_h, keep_w)
        else:
            keep_h = random.randint(2, 4 - (1 if amount == 2 else 0))
            keep_w = random.randint(2, 3)
            full_shape = (keep_h + amount, keep_w)
            keep_shape = (keep_h, keep_w)
        if keep_shape in used_shapes:
            continue

        max_area = min(6, full_shape[0] * full_shape[1])
        for _ in range(500):
            area = random.randint(4, max_area)
            full_coords = _random_connected_shape(full_shape[0], full_shape[1], area)
            if full_coords.size == 0 or not _spans_shape(full_coords, full_shape):
                continue

            part_local = _edge_subset(full_coords, full_shape, direction)
            if part_local is None:
                continue

            cropped = _crop_local_coords(full_coords, full_shape, crop_side, amount)
            if cropped is None:
                continue
            cropped_coords, cropped_shape = cropped
            if tuple(map(int, cropped_shape)) != keep_shape:
                continue
            if not _spans_shape(cropped_coords, keep_shape):
                continue
            if cropped_coords.shape[0] < 3:
                continue

            wrong = _crop_local_coords(full_coords, full_shape, OTHER_AXIS_SIDE[crop_side], amount)
            if wrong is None:
                continue
            wrong_coords, wrong_shape = wrong
            if tuple(map(int, wrong_shape)) != keep_shape:
                continue
            correct_patch = _render_local_coords(cropped_coords, keep_shape, 1)
            wrong_patch = _render_local_coords(wrong_coords, keep_shape, 1)
            if np.array_equal(correct_patch, wrong_patch):
                continue

            trial_part = part_local + np.array([step[0] * 4, step[1] * 4], dtype=int)
            try:
                best = _best_cover_translation(full_coords, trial_part)
            except AssertionError:
                continue
            if _vector_direction(*best) != direction:
                continue

            return {
                "color": int(color),
                "direction": direction,
                "crop_side": crop_side,
                "crop_amount": int(amount),
                "full_shape": tuple(map(int, full_shape)),
                "keep_shape": tuple(map(int, keep_shape)),
                "full_coords": full_coords,
                "part_local": part_local,
                "cropped_coords": cropped_coords,
                "patch": _render_local_coords(cropped_coords, keep_shape, color),
            }
    return None


def _pack_cavities(specs: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    min_h = max(8, max(spec["keep_shape"][0] for spec in specs) + 4)
    min_w = max(9, max(spec["keep_shape"][1] for spec in specs) + 5)
    for _ in range(200):
        frame_h = random.randint(min_h, 12)
        frame_w = random.randint(min_w, 13)
        order = list(range(len(specs)))
        random.shuffle(order)
        blocked = np.zeros((frame_h, frame_w), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(specs)
        ok = True
        for idx in order:
            h, w = specs[idx]["keep_shape"]
            cands: list[tuple[int, int]] = []
            for r0 in range(1, frame_h - h):
                for c0 in range(1, frame_w - w):
                    rs = slice(max(0, r0 - 1), min(frame_h, r0 + h + 1))
                    cs = slice(max(0, c0 - 1), min(frame_w, c0 + w + 1))
                    if blocked[rs, cs].any():
                        continue
                    cands.append((r0, c0))
            if not cands:
                ok = False
                break
            r0, c0 = random.choice(cands)
            positions[idx] = (r0, c0)
            blocked[max(0, r0 - 1) : min(frame_h, r0 + h + 1), max(0, c0 - 1) : min(frame_w, c0 + w + 1)] = True
        if ok:
            return frame_h, frame_w, [(int(r), int(c)) for r, c in positions if r is not None and c is not None]
    return None


def _margin_cells(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _place_external_pairs(
    grid: np.ndarray,
    specs: list[dict],
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    forbidden = {
        (r, c)
        for r in range(max(0, fr0 - 1), min(h, fr1 + 2))
        for c in range(max(0, fc0 - 1), min(w, fc1 + 2))
    }
    occupied_margin: set[tuple[int, int]] = set()

    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        spec = specs[idx]
        step = DIR_STEP[spec["direction"]]
        fh, fw = spec["full_shape"]
        origins = [(r0, c0) for r0 in range(h - fh + 1) for c0 in range(w - fw + 1)]
        random.shuffle(origins)
        placed = False
        for dist in random.sample([3, 4, 5], k=3):
            shift = np.array([step[0] * dist, step[1] * dist], dtype=int)
            for top, left in origins:
                offset = np.array([top, left], dtype=int)
                full_abs = spec["full_coords"] + offset
                part_abs = spec["part_local"] + offset + shift
                all_abs = np.vstack([full_abs, part_abs])
                if (
                    all_abs[:, 0].min() < 0
                    or all_abs[:, 1].min() < 0
                    or all_abs[:, 0].max() >= h
                    or all_abs[:, 1].max() >= w
                ):
                    continue
                all_set = {tuple(map(int, rc)) for rc in all_abs}
                if all_set & forbidden:
                    continue
                if all_set & occupied_margin:
                    continue

                spec["distance"] = int(dist)
                spec["full_abs"] = full_abs
                spec["part_abs"] = part_abs
                grid[full_abs[:, 0], full_abs[:, 1]] = spec["color"]
                grid[part_abs[:, 0], part_abs[:, 1]] = spec["color"]
                occupied_margin |= _margin_cells(all_abs, (h, w))
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _validate_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    specs: list[dict],
    cavities: list[dict],
) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    seen_colors = Counter(int(v) for v in grid.reshape(-1) if v not in (BACKGROUND, FRAME_COLOR))
    for spec in specs:
        if seen_colors[spec["color"]] != spec["full_coords"].shape[0] + spec["part_local"].shape[0]:
            return False

        try:
            best = _best_cover_translation(spec["full_abs"], spec["part_abs"])
        except AssertionError:
            return False
        if _vector_direction(*best) != spec["direction"]:
            return False

        wrong = _crop_local_coords(
            spec["full_coords"],
            spec["full_shape"],
            OTHER_AXIS_SIDE[spec["crop_side"]],
            spec["crop_amount"],
        )
        if wrong is None:
            return False
        wrong_patch = _render_local_coords(wrong[0], spec["keep_shape"], 1)
        if np.array_equal(wrong_patch, _render_local_coords(spec["cropped_coords"], spec["keep_shape"], 1)):
            return False

        matches = []
        for idx, cavity in enumerate(cavities):
            ch, cw = cavity["shape"]
            if spec["crop_side"] in ("left", "right"):
                if ch != spec["full_shape"][0]:
                    continue
                amount = spec["full_shape"][1] - cw
            else:
                if cw != spec["full_shape"][1]:
                    continue
                amount = spec["full_shape"][0] - ch
            if amount not in (1, 2):
                continue
            cropped = _crop_local_coords(spec["full_coords"], spec["full_shape"], spec["crop_side"], amount)
            if cropped is None:
                continue
            cropped_coords, shape = cropped
            if tuple(map(int, shape)) != tuple(map(int, cavity["shape"])):
                continue
            if not _spans_shape(cropped_coords, cavity["shape"]):
                continue
            matches.append(idx)
        if matches != [spec["cavity_index"]]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.choice([2, 3])
    directions = random.sample(DIRECTIONS, k=sprite_count)
    colors = random.sample(SPRITE_COLORS, k=sprite_count)

    specs: list[dict] = []
    used_shapes: set[tuple[int, int]] = set()
    for direction, color in zip(directions, colors):
        spec = _generate_sprite_spec(direction, color, used_shapes)
        if spec is None:
            return None
        used_shapes.add(spec["keep_shape"])
        specs.append(spec)

    packed = _pack_cavities(specs)
    if packed is None:
        return None
    frame_h, frame_w, positions = packed
    if len(positions) != len(specs):
        return None

    frame = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
    cavities: list[dict] = []
    for cavity_index, spec in enumerate(specs):
        r0, c0 = positions[cavity_index]
        ch, cw = spec["keep_shape"]
        frame[r0 : r0 + ch, c0 : c0 + cw] = BACKGROUND
        spec["cavity_index"] = cavity_index
        spec["cavity_pos"] = (r0, c0)
        cavities.append({"bbox": (r0, c0, r0 + ch - 1, c0 + cw - 1), "shape": spec["keep_shape"]})

    expected = frame.copy()
    for spec in specs:
        rr, cc = spec["cavity_pos"]
        _paste(expected, spec["patch"], rr, cc, mask=spec["patch"] != BACKGROUND)

    grid_h = random.randint(max(18, frame_h + 8), min(26, frame_h + 14))
    grid_w = random.randint(max(18, frame_w + 8), min(26, frame_w + 14))
    if grid_h > 30 or grid_w > 30:
        return None
    frame_top = random.randint(3, grid_h - frame_h - 3)
    frame_left = random.randint(3, grid_w - frame_w - 3)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = FRAME_COLOR
    for spec in specs:
        rr, cc = spec["cavity_pos"]
        ch, cw = spec["keep_shape"]
        grid[frame_top + rr : frame_top + rr + ch, frame_left + cc : frame_left + cc + cw] = BACKGROUND

    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
    if not _place_external_pairs(grid, specs, frame_bbox):
        return None

    if not _validate_semantics(grid, expected, specs, cavities):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
