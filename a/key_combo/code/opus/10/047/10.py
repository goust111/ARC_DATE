from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
TRACK = 2
SPRITE_COLORS = (1, 3, 4, 7)

GEN_RETRIES = 400
GRID_SIDE_RANGE = (16, 20)
OBJECT_COUNT_CHOICES = (3, 4)
SPRITE_AREA_RANGE = (5, 8)
SPRITE_BOX_RANGE = (3, 4)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neighbors = _neighbors4 if int(connectivity) == 4 else _neighbors8
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _transform_patch(patch: np.ndarray, key: str) -> np.ndarray:
    if key == "horizontal_track":
        return np.flipud(patch)
    if key == "vertical_track":
        return np.fliplr(patch)
    if key == "diagonal_track":
        return np.rot90(patch, 2)
    raise ValueError(f"unknown track kind: {key}")


def _track_kind(coords: list[tuple[int, int]]) -> str:
    if len(coords) != 3:
        raise AssertionError("each red trajectory must have exactly 3 pixels")
    pts = sorted((int(r), int(c)) for r, c in coords)
    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    if len(rows) == 1:
        return "horizontal_track"
    if len(cols) == 1:
        return "vertical_track"
    if len(rows) == 3 and len(cols) == 3:
        s1 = {r - c for r, c in pts}
        s2 = {r + c for r, c in pts}
        if len(s1) == 1 or len(s2) == 1:
            rows_sorted = sorted(rows)
            cols_sorted = sorted(cols)
            if rows_sorted[-1] - rows_sorted[0] == 2 and cols_sorted[-1] - cols_sorted[0] == 2:
                return "diagonal_track"
    raise AssertionError("red trajectory must be horizontal, vertical, or 45-degree diagonal")


def _touches_8(comp_a: list[tuple[int, int]], comp_b: list[tuple[int, int]]) -> bool:
    bset = set(comp_b)
    for r, c in comp_a:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    sprite_mask = (g != BG) & (g != TRACK)
    track_mask = g == TRACK

    sprites = []
    for coords in _find_components(sprite_mask, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = {int(g[r, c]) for r, c in coords}
        sprites.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "colors": colors,
            }
        )

    tracks = []
    for coords in _find_components(track_mask, connectivity=8):
        tracks.append({"coords": coords, "kind": _track_kind(coords)})

    matches: list[tuple[int, int]] = []
    for si, sprite in enumerate(sprites):
        adjacent = [ti for ti, track in enumerate(tracks) if _touches_8(sprite["coords"], track["coords"])]
        if len(adjacent) != 1:
            raise AssertionError("each sprite must have exactly one adjacent red trajectory")
        matches.append((si, adjacent[0]))

    if len({ti for _, ti in matches}) != len(tracks):
        raise AssertionError("each trajectory must belong to exactly one sprite")

    return sprites, tracks, matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror sprites in place using adjacent red trajectories as reflection-type codes.

    Input structure:
    - Black (0) background contains 3-4 colored 4-connected sprite bodies and several red (2) 3-pixel trajectories.
    - Each non-black non-red sprite has exactly one red straight-line trajectory in its 8-neighborhood.
    - A trajectory is horizontal, vertical, or 45-degree diagonal, and one endpoint touches its sprite by 8-connectivity.

    Rule:
    - Identify each sprite body together with its adjacent red trajectory.
    - Horizontal trajectory -> flip the sprite across its bbox horizontal midline; vertical trajectory -> flip across its bbox vertical midline; diagonal trajectory -> rotate the sprite 180 degrees about its bbox center.
    - Write the transformed sprite back into the same bbox position, replacing the original body.
    - Erase all red trajectory pixels to black and leave everything else unchanged.

    Key insight:
    - The red trajectory does not show the mirror axis directly; its geometric direction orthogonally encodes which in-bbox reflection to apply, and that cue disappears after use.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sprites, tracks, matches = _parse_scene(g)

    out[out == TRACK] = BG
    for si, ti in matches:
        sprite = sprites[si]
        track = tracks[ti]
        r0, c0, r1, c1 = sprite["bbox"]
        transformed = _transform_patch(sprite["patch"], track["kind"])
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed
    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(*SPRITE_BOX_RANGE)
        w = rng.randint(*SPRITE_BOX_RANGE)
        target = rng.randint(*SPRITE_AREA_RANGE)
        if target > h * w:
            continue

        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != target:
            continue

        coords = list(cells)
        mask = _tight_mask(coords)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue

        variants = {
            mask.tobytes(),
            np.flipud(mask).tobytes(),
            np.fliplr(mask).tobytes(),
            np.rot90(mask, 2).tobytes(),
        }
        if len(variants) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate sprite mask")


def _line_steps(kind: str):
    if kind == "horizontal_track":
        return [(0, 0), (0, 1), (0, 2)]
    if kind == "vertical_track":
        return [(0, 0), (1, 0), (2, 0)]
    return [[(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]


def _track_candidates(mask: np.ndarray, top: int, left: int, zone_h: int, zone_w: int, kind: str):
    sprite_coords = {(top + r, left + c) for r, c in np.argwhere(mask)}
    candidates: list[list[tuple[int, int]]] = []
    step_sets = _line_steps(kind)
    if kind != "diagonal_track":
        step_sets = [step_sets]

    for steps in step_sets:
        max_r = max(dr for dr, _ in steps)
        max_c = max(dc for _, dc in steps)
        for r0 in range(zone_h - max_r):
            for c0 in range(zone_w - max_c):
                coords = [(r0 + dr, c0 + dc) for dr, dc in steps]
                if any((r, c) in sprite_coords for r, c in coords):
                    continue
                adjacent = []
                for idx, (r, c) in enumerate(coords):
                    hit = False
                    for nr, nc in _neighbors8(r, c):
                        if (nr, nc) in sprite_coords:
                            hit = True
                            break
                    if hit:
                        adjacent.append(idx)
                if adjacent in ([0], [2]):
                    candidates.append(coords)
    return candidates


def _build_group(zone_h: int, zone_w: int, color: int, kind: str, rng: random.Random):
    for _ in range(200):
        mask = _random_connected_mask(rng)
        h, w = mask.shape
        positions = [(r, c) for r in range(zone_h - h + 1) for c in range(zone_w - w + 1)]
        rng.shuffle(positions)
        for top, left in positions:
            tracks = _track_candidates(mask, top, left, zone_h, zone_w, kind)
            if not tracks:
                continue
            track = rng.choice(tracks)
            inp = np.zeros((zone_h, zone_w), dtype=int)
            inp[top : top + h, left : left + w][mask] = int(color)
            for r, c in track:
                inp[r, c] = TRACK

            patch = np.zeros((h, w), dtype=int)
            patch[mask] = int(color)
            out = np.zeros((zone_h, zone_w), dtype=int)
            out[top : top + h, left : left + w] = _transform_patch(patch, kind)

            return {
                "input": inp,
                "output": out,
                "mask": mask,
                "bbox": (top, left, top + h - 1, left + w - 1),
                "track": track,
                "kind": kind,
                "color": int(color),
            }
    return None


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    if row_gap == 0:
        return col_gap
    if col_gap == 0:
        return row_gap
    return min(row_gap, col_gap)


def _orientation_plan(count: int, rng: random.Random) -> list[str]:
    plan = ["horizontal_track", "vertical_track", "diagonal_track"]
    if count == 4:
        plan.append(rng.choice(plan))
    rng.shuffle(plan)
    return plan


def _zone_slices(H: int, W: int):
    mid_r = H // 2
    mid_c = W // 2
    return [
        (slice(0, mid_r), slice(0, mid_c)),
        (slice(0, mid_r), slice(mid_c, W)),
        (slice(mid_r, H), slice(0, mid_c)),
        (slice(mid_r, H), slice(mid_c, W)),
    ]


def _validate_generated_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        sprites, tracks, matches = _parse_scene(inp)
    except AssertionError:
        return False

    if len(sprites) not in OBJECT_COUNT_CHOICES or len(tracks) != len(sprites):
        return False
    if any(len(sprite["colors"]) != 1 for sprite in sprites):
        return False

    kinds = {tracks[ti]["kind"] for _, ti in matches}
    if kinds != {"horizontal_track", "vertical_track", "diagonal_track"}:
        return False

    for sprite in sprites:
        mask = _tight_mask(sprite["coords"])
        variants = {
            mask.tobytes(),
            np.flipud(mask).tobytes(),
            np.fliplr(mask).tobytes(),
            np.rot90(mask, 2).tobytes(),
        }
        if len(variants) != 4:
            return False

    bboxes = [sprite["bbox"] for sprite in sprites]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _bbox_gap(bboxes[i], bboxes[j]) < 4:
                return False

    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        H = rng.randint(*GRID_SIDE_RANGE)
        W = rng.randint(*GRID_SIDE_RANGE)
        count = rng.choice(OBJECT_COUNT_CHOICES)
        zones = _zone_slices(H, W)
        chosen_zone_ids = rng.sample(range(4), count)
        colors = rng.sample(list(SPRITE_COLORS), count)
        kinds = _orientation_plan(count, rng)

        inp = np.zeros((H, W), dtype=int)
        expected = np.zeros((H, W), dtype=int)
        metas = []
        ok = True
        for zone_id, color, kind in zip(chosen_zone_ids, colors, kinds):
            rs, cs = zones[zone_id]
            group = _build_group(rs.stop - rs.start, cs.stop - cs.start, color, kind, rng)
            if group is None:
                ok = False
                break
            inp[rs, cs] = group["input"]
            expected[rs, cs] = group["output"]
            top, left, bottom, right = group["bbox"]
            metas.append((top + rs.start, left + cs.start, bottom + rs.start, right + cs.start))

        if not ok:
            continue

        if any(_bbox_gap(metas[i], metas[j]) < 4 for i in range(len(metas)) for j in range(i + 1, len(metas))):
            continue
        if _validate_generated_instance(inp, expected):
            return inp

    raise RuntimeError("generate_input failed to produce a valid sample")
