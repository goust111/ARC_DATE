from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEED = 5
FILL = 7
OBJECT_COLORS = [1, 2, 3, 4, 6, 8, 9]

GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (15, 20)
OBJECT_COUNT_RANGE = (3, 5)
NOISE_COUNT_RANGE = (1, 3)
MIN_OBJECT_AREA = 5
MAX_OBJECT_AREA = 12


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _anti_diag_reflect(mask: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(mask)).T


def _raw_symmetry(mask: np.ndarray) -> str | None:
    horiz = bool(np.array_equal(mask, np.flipud(mask)))
    vert = bool(np.array_equal(mask, np.fliplr(mask)))
    diag_main = bool(mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T))
    diag_anti = bool(mask.shape[0] == mask.shape[1] and np.array_equal(mask, _anti_diag_reflect(mask)))
    diag_count = int(diag_main) + int(diag_anti)

    if horiz and not vert and diag_count == 0:
        return "horizontal"
    if vert and not horiz and diag_count == 0:
        return "vertical"
    if not horiz and not vert and diag_count == 1:
        return "diag_main" if diag_main else "diag_anti"
    return None


def _move_dirs(symmetry: str) -> tuple[tuple[int, int], ...]:
    if symmetry == "horizontal":
        return ((0, -1), (0, 1))
    if symmetry == "vertical":
        return ((-1, 0), (1, 0))
    if symmetry in {"diag_main", "diag_anti"}:
        return ((-1, -1), (-1, 1), (1, -1), (1, 1))
    raise ValueError(f"unknown symmetry {symmetry}")


def _seed_off_axis(seed: tuple[int, int], symmetry: str, shape: tuple[int, int]) -> bool:
    r, c = seed
    h, w = shape
    if symmetry == "horizontal":
        return not (h % 2 == 1 and r == h // 2)
    if symmetry == "vertical":
        return not (w % 2 == 1 and c == w // 2)
    if symmetry == "diag_main":
        return r != c
    if symmetry == "diag_anti":
        return r + c != h - 1
    raise ValueError(symmetry)


def _reachable_local(mask: np.ndarray, seed: tuple[int, int], symmetry: str) -> set[tuple[int, int]]:
    h, w = mask.shape
    dirs = _move_dirs(symmetry)
    q = deque([seed])
    seen = {seed}
    reached: set[tuple[int, int]] = set()
    while q:
        r, c = q.popleft()
        if mask[r, c]:
            reached.add((r, c))
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if mask[nr, nc] or (nr, nc) == seed:
                seen.add((nr, nc))
                q.append((nr, nc))
    return reached


def _transform_mask(mask: np.ndarray, seed: tuple[int, int], op: str) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = mask.shape
    r, c = seed
    if op == "id":
        return mask.copy(), (r, c)
    if op == "flip_lr":
        return np.fliplr(mask), (r, w - 1 - c)
    if op == "flip_ud":
        return np.flipud(mask), (h - 1 - r, c)
    if op == "rot180":
        return np.rot90(mask, 2), (h - 1 - r, w - 1 - c)
    raise ValueError(op)


def _build_horizontal_library() -> list[dict]:
    library: list[dict] = []
    for h in range(3, 6):
        half = (h + 1) // 2
        for w in range(3, 6):
            for bits in range(1 << (half * w)):
                top = np.array([(bits >> i) & 1 for i in range(half * w)], dtype=bool).reshape(half, w)
                if h % 2 == 0:
                    mask = np.vstack([top, top[::-1]])
                else:
                    mask = np.vstack([top, top[-2::-1]])
                if not mask.any():
                    continue
                mask = _crop_mask(mask)
                area = int(mask.sum())
                if not (MIN_OBJECT_AREA <= area <= MAX_OBJECT_AREA):
                    continue
                if len(_components_mask(mask)) != 1:
                    continue
                if _raw_symmetry(mask) != "horizontal":
                    continue
                seeds = []
                hh, ww = mask.shape
                for r in range(hh):
                    for c in range(ww):
                        if mask[r, c]:
                            continue
                        seed = (r, c)
                        if not _seed_off_axis(seed, "horizontal", mask.shape):
                            continue
                        reached = _reachable_local(mask, seed, "horizontal")
                        if 2 <= len(reached) < area:
                            seeds.append(seed)
                if len(seeds) >= 2:
                    library.append({"mask": mask, "symmetry": "horizontal", "seeds": seeds})
    assert library
    return library


def _build_diagonal_library() -> list[dict]:
    library: list[dict] = []
    for n in range(3, 6):
        tri = n * (n + 1) // 2
        positions = [(r, c) for r in range(n) for c in range(r, n)]
        for bits in range(1 << tri):
            mask = np.zeros((n, n), dtype=bool)
            for i, (r, c) in enumerate(positions):
                if (bits >> i) & 1:
                    mask[r, c] = True
                    mask[c, r] = True
            if not mask.any():
                continue
            mask = _crop_mask(mask)
            area = int(mask.sum())
            if not (MIN_OBJECT_AREA <= area <= MAX_OBJECT_AREA):
                continue
            if len(_components_mask(mask)) != 1:
                continue
            if _raw_symmetry(mask) != "diag_main":
                continue
            seeds = []
            hh, ww = mask.shape
            for r in range(hh):
                for c in range(ww):
                    if mask[r, c]:
                        continue
                    seed = (r, c)
                    if not _seed_off_axis(seed, "diag_main", mask.shape):
                        continue
                    reached = _reachable_local(mask, seed, "diag_main")
                    if 2 <= len(reached) < area:
                        seeds.append(seed)
                if len(seeds) >= 2:
                    break
            if len(seeds) >= 2:
                library.append({"mask": mask, "symmetry": "diag_main", "seeds": seeds})
    assert library
    return library


HORIZONTAL_LIBRARY = _build_horizontal_library()
VERTICAL_LIBRARY = [
    {"mask": entry["mask"].T.copy(), "symmetry": "vertical", "seeds": [(c, r) for r, c in entry["seeds"]]}
    for entry in HORIZONTAL_LIBRARY
]
DIAGONAL_LIBRARY = _build_diagonal_library()


def _instantiate_template(symmetry: str) -> dict:
    if symmetry == "horizontal":
        entry = random.choice(HORIZONTAL_LIBRARY)
        op = random.choice(["id", "flip_lr", "flip_ud", "rot180"])
    elif symmetry == "vertical":
        entry = random.choice(VERTICAL_LIBRARY)
        op = random.choice(["id", "flip_lr", "flip_ud", "rot180"])
    elif symmetry == "diagonal":
        entry = random.choice(DIAGONAL_LIBRARY)
        op = random.choice(["id", "rot180", "flip_lr", "flip_ud"])
    else:
        raise ValueError(symmetry)

    seed = random.choice(entry["seeds"])
    mask, seed = _transform_mask(entry["mask"], seed, op)
    raw = _raw_symmetry(mask)
    assert raw is not None
    return {"mask": mask, "seed": seed, "raw_symmetry": raw}


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    color: int,
) -> tuple[int, int] | None:
    h, w = mask.shape
    gh, gw = grid.shape
    candidates = [(r0, c0) for r0 in range(gh - h + 1) for c0 in range(gw - w + 1)]
    random.shuffle(candidates)

    fg_local = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    fg_local.add((int(seed[0]), int(seed[1])))
    for r0, c0 in candidates:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(gh, r0 + h + 1)
        cc1 = min(gw, c0 + w + 1)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        ok = True
        for dr, dc in fg_local:
            rr = r0 + dr
            cc = c0 + dc
            for nr in range(rr - 1, rr + 2):
                for nc in range(cc - 1, cc + 2):
                    if 0 <= nr < gh and 0 <= nc < gw and occupied[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue

        for dr, dc in np.argwhere(mask):
            grid[r0 + int(dr), c0 + int(dc)] = int(color)
            occupied[r0 + int(dr), c0 + int(dc)] = True
        sr, sc = seed
        grid[r0 + sr, c0 + sc] = SEED
        occupied[r0 + sr, c0 + sc] = True
        reserved[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _place_noise(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved: np.ndarray,
    color: int,
) -> tuple[int, int] | None:
    gh, gw = grid.shape
    candidates = [(r, c) for r in range(gh) for c in range(gw)]
    random.shuffle(candidates)
    for r, c in candidates:
        if reserved[r, c]:
            continue
        ok = True
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < gh and 0 <= nc < gw and occupied[nr, nc]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[r, c] = int(color)
            occupied[r, c] = True
            return r, c
    return None


def _expected_from_metadata(inp: np.ndarray, metadata: dict) -> np.ndarray:
    out = inp.copy()
    for r, c in metadata["noise"]:
        out[r, c] = BG

    for obj in metadata["objects"]:
        r0 = obj["origin"][0]
        c0 = obj["origin"][1]
        mask = obj["mask"]
        seed = obj["seed"]
        symmetry = obj["raw_symmetry"]
        reached = _reachable_local(mask, seed, symmetry)
        for dr, dc in np.argwhere(mask):
            rr = r0 + int(dr)
            cc = c0 + int(dc)
            if (int(dr), int(dc)) in reached:
                out[rr, cc] = FILL
        out[r0 + seed[0], c0 + seed[1]] = FILL
    return out


def _validate_generated_scene(inp: np.ndarray, metadata: dict) -> None:
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert len(metadata["objects"]) == metadata["count"]
    colors = [obj["color"] for obj in metadata["objects"]]
    assert len(colors) == len(set(colors))
    raw_types = {obj["raw_symmetry"] for obj in metadata["objects"]}
    assert {"horizontal", "vertical"} <= raw_types
    assert any(t in {"diag_main", "diag_anti"} for t in raw_types)

    non_seed = (inp != BG) & (inp != SEED)
    comps = []
    for color in sorted(int(v) for v in np.unique(inp) if v not in (BG, SEED)):
        color_mask = inp == color
        for comp in _components_mask(color_mask):
            comps.append((color, comp))

    object_areas = sorted(obj["mask"].sum() for obj in metadata["objects"])
    seen_object_areas = sorted(int(comp.shape[0]) for _color, comp in comps if int(comp.shape[0]) > 1)
    assert seen_object_areas == object_areas
    assert sum(1 for _color, comp in comps if int(comp.shape[0]) == 1) == len(metadata["noise"])

    for obj in metadata["objects"]:
        r0, c0 = obj["origin"]
        mask = obj["mask"]
        seed = obj["seed"]
        raw = obj["raw_symmetry"]
        sr, sc = seed
        assert inp[r0 + sr, c0 + sc] == SEED
        assert _seed_off_axis(seed, raw, mask.shape)
        assert _raw_symmetry(mask) == raw
        reached = _reachable_local(mask, seed, raw)
        assert 2 <= len(reached) < int(mask.sum())

    expected = _expected_from_metadata(inp, metadata)
    actual = main(inp)
    assert np.array_equal(actual, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each symmetric object only along the fill directions allowed by its reflection type.

    Input structure:
    - The background is black.
    - Several non-grey monochrome 4-connected objects are separated from one another, and isolated same-color singletons act as noise.
    - Each real object has exactly one grey seed cell inside its bbox but outside the object pixels, and the object is uniquely horizontal-, vertical-, or diagonal-reflection symmetric.

    Rule:
    - Remove every non-grey singleton as noise.
    - For each remaining object, detect whether it is horizontal-, vertical-, or diagonal-symmetric.
    - Starting from that object's grey seed, flood only through the object's cells using left/right moves for horizontal symmetry, up/down moves for vertical symmetry, and the four diagonal moves for diagonal symmetry.
    - Recolor every reached object cell, plus the seed itself, to orange; leave unreached object cells unchanged.

    Key insight:
    The symmetry class does not choose a color directly; it gates which movement directions are legal, so the final orange region is a direction-restricted reachability slice of the object.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if v not in (BG, SEED)):
        mask = g == color
        for comp in _components_mask(mask):
            area = int(comp.shape[0])
            if area == 1:
                r, c = map(int, comp[0])
                out[r, c] = BG
                continue

            r0, c0, r1, c1 = _bbox(comp)
            local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in comp:
                local_mask[int(rr) - r0, int(cc) - c0] = True
            symmetry = _raw_symmetry(local_mask)
            assert symmetry is not None, "object must have exactly one supported reflection type"

            local_seed_cells = np.argwhere(g[r0 : r1 + 1, c0 : c1 + 1] == SEED)
            assert local_seed_cells.shape[0] == 1, "each object bbox must contain exactly one seed"
            sr, sc = map(int, local_seed_cells[0])
            seed = (sr, sc)
            assert not local_mask[sr, sc]
            assert _seed_off_axis(seed, symmetry, local_mask.shape)

            objects.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "mask": local_mask,
                    "seed": seed,
                    "symmetry": symmetry,
                }
            )

    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        reached = _reachable_local(obj["mask"], obj["seed"], obj["symmetry"])
        for dr, dc in reached:
            out[r0 + dr, c0 + dc] = FILL
        sr, sc = obj["seed"]
        out[r0 + sr, c0 + sc] = FILL

    return out


def _generate_attempt() -> tuple[np.ndarray, dict] | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)

    base_types = ["horizontal", "vertical", "diagonal"]
    extra = [random.choice(base_types) for _ in range(count - 3)]
    requested_types = base_types + extra
    random.shuffle(requested_types)
    colors = random.sample(OBJECT_COLORS, count)

    objects = []
    for symmetry, color in zip(requested_types, colors):
        template = _instantiate_template(symmetry)
        origin = _place_patch(grid, occupied, reserved, template["mask"], template["seed"], color)
        if origin is None:
            return None
        objects.append(
            {
                "origin": origin,
                "mask": template["mask"],
                "seed": template["seed"],
                "raw_symmetry": template["raw_symmetry"],
                "color": color,
            }
        )

    noise_count = random.randint(*NOISE_COUNT_RANGE)
    noise = []
    noise_colors = [random.choice(colors) for _ in range(noise_count)]
    for color in noise_colors:
        pos = _place_noise(grid, occupied, reserved, color)
        if pos is None:
            return None
        noise.append(pos)

    metadata = {"objects": objects, "noise": noise, "count": count}
    _validate_generated_scene(grid, metadata)
    return grid, metadata


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid symmetry-gated flood-fill puzzle")
