from __future__ import annotations

import itertools
import random
from functools import lru_cache

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

CONNECT4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBOR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GRID_SIDE_RANGE = (19, 26)
RADIUS_RANGE = (5, 8)
SPRITE_COUNT_RANGE = (4, 7)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in CONNECT4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(sorted(pts), dtype=int))
    return out


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    stack = [next(iter(coords))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in CONNECT4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _normalize(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _shift(coords: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + r0, c + c0) for r, c in coords)


def _diamond_outline(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cr, cc = center
    return {
        (r, c)
        for r in range(cr - radius, cr + radius + 1)
        for c in range(cc - radius, cc + radius + 1)
        if abs(r - cr) + abs(c - cc) == radius
    }


def _classify(coords: tuple[tuple[int, int], ...], center: tuple[int, int], radius: int) -> str:
    cr, cc = center
    dists = [abs(r - cr) + abs(c - cc) for r, c in coords]
    if all(d < radius for d in dists):
        return "inside"
    if all(d > radius for d in dists):
        return "outside"
    return "cross"


def _infer_completed_sprite(
    comp: np.ndarray, grid_shape: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...], int]:
    h, w = map(int, grid_shape)
    visible = {tuple(map(int, rc)) for rc in comp}
    candidates: list[tuple[tuple[tuple[int, int], ...], int]] = []

    for axis2 in range(0, 2 * w - 1):
        full = set(visible)
        valid = True
        for r, c in visible:
            mc = axis2 - c
            if not (0 <= mc < w):
                valid = False
                break
            if 2 * c > axis2 and (r, mc) not in visible:
                valid = False
                break
            full.add((r, mc))
        if not valid:
            continue
        added = full - visible
        if not (1 <= len(added) <= 3):
            continue
        if any(2 * c <= axis2 for _, c in added):
            continue
        if not (3 <= len(full) <= 7):
            continue
        if not _is_connected(full):
            continue
        coords = tuple(sorted(full))
        candidates.append((coords, axis2))

    unique = {}
    for coords, axis2 in candidates:
        unique.setdefault(coords, []).append(axis2)
    if len(unique) != 1:
        raise AssertionError("sprite axis is ambiguous")
    coords = next(iter(unique))
    axis2 = unique[coords][0]
    return coords, int(axis2)


def _parse_diamond(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    red = np.argwhere(grid == RED)
    assert red.size > 0, "expected a red diamond outline"
    r0, c0, r1, c1 = _bbox(red)
    assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "diamond bbox must have odd side lengths"
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    radius = (r1 - r0) // 2
    assert radius >= 2 and radius == (c1 - c0) // 2, "diamond must be centered and isotropic"
    cr, cc = center
    assert all(abs(int(r) - cr) + abs(int(c) - cc) == radius for r, c in red), "red pixels must form a diamond outline"
    return center, int(radius)


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, list[dict], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    center, radius = _parse_diamond(g)
    shrunk = radius - 1
    assert shrunk >= 1

    comps = _components(g == GREY)
    assert comps, "expected gray sprites"

    out = np.full_like(g, BLACK)
    for r, c in _diamond_outline(center, shrunk):
        out[r, c] = RED

    meta = []
    for comp in comps:
        full, axis2 = _infer_completed_sprite(comp, g.shape)
        cls = _classify(full, center, shrunk)
        if cls == "inside":
            for r, c in full:
                out[r, c] = BLUE
        elif cls == "outside":
            for r, c in full:
                out[r, c] = GREEN
        meta.append(
            {
                "visible": tuple(sorted(map(tuple, comp.tolist()))),
                "full": full,
                "axis2": axis2,
                "class_shrunk": cls,
                "class_visible_shrunk": _classify(tuple(sorted(map(tuple, comp.tolist()))), center, shrunk),
                "class_full_original": _classify(full, center, radius),
            }
        )
    return out, meta, center, radius


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each half-erased gray mirror sprite, shrink the red diamond by one layer, then recolor or erase sprites by their post-transform position.

    Input structure:
    - A black background contains one complete red Manhattan-diamond outline near the center.
    - Several gray sprites are scattered around it; each sprite is a 4-connected left-right symmetric shape whose right half is partially erased.
    - Every sprite is small, disconnected from the others, and can be completed by mirroring its visible left side across a vertical axis.

    Rule:
    - For each gray sprite, infer its vertical mirror axis and restore the missing right-side pixels to obtain the full symmetric shape.
    - Replace the original red diamond outline by a new outline with the same center and radius reduced by one; its interior is the set of cells with strictly smaller Manhattan distance.
    - Classify each completed sprite against that shrunk interior: fully inside becomes blue, fully outside the interior and outline becomes green, and every sprite touching both sides of the boundary is erased.

    Key insight:
    - The classification is only correct after two independent geometry changes are applied: the sprite grows by mirror completion while the reference diamond shrinks, so either skipped step can change the final class.
    """
    return _solve_with_meta(grid)[0]


@lru_cache(maxsize=1)
def _shape_catalog() -> tuple[dict, ...]:
    catalog = {}
    for height in range(1, 6):
        for width in range(2, 7):
            orbits = []
            for r in range(height):
                for c in range(width):
                    mc = width - 1 - c
                    if c > mc:
                        continue
                    if c == mc:
                        orbit = ((r, c),)
                    else:
                        orbit = tuple(sorted(((r, c), (r, mc))))
                    orbits.append(orbit)
            for mask in range(1, 1 << len(orbits)):
                full = set()
                for i, orbit in enumerate(orbits):
                    if mask & (1 << i):
                        full.update(orbit)
                if not (3 <= len(full) <= 7):
                    continue
                if not _is_connected(full):
                    continue
                norm = _normalize(full)
                fh = max(r for r, _ in norm) + 1
                fw = max(c for _, c in norm) + 1
                axis2 = fw - 1
                right = [pt for pt in norm if 2 * pt[1] > axis2]
                if len(right) < 2:
                    continue
                variants = []
                for k in range(1, min(3, len(right) - 1) + 1):
                    for removed in itertools.combinations(right, k):
                        visible = set(norm) - set(removed)
                        if not _is_connected(visible):
                            continue
                        try:
                            inferred, inferred_axis2 = _infer_completed_sprite(
                                np.array(sorted(visible), dtype=int), (fh, fw)
                            )
                        except AssertionError:
                            continue
                        if inferred != norm or inferred_axis2 != axis2:
                            continue
                        variants.append(
                            {
                                "full": norm,
                                "visible": tuple(sorted(visible)),
                                "height": fh,
                                "width": fw,
                                "removed": tuple(sorted(removed)),
                            }
                        )
                if variants:
                    catalog.setdefault(norm, variants)
    all_variants = [random.choice(v) for v in catalog.values()]
    return tuple(sorted(all_variants, key=lambda item: (len(item["full"]), item["height"], item["width"], item["full"])))


def _sprite_neighborhood(coords: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    out = set(coords)
    for r, c in coords:
        for dr, dc in NEIGHBOR8:
            out.add((r + dr, c + dc))
    return out


def _candidate_placements(
    shape: dict,
    grid_shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    desired: str,
    occupied_full: set[tuple[int, int]],
    blocked_visible: set[tuple[int, int]],
    forbid: set[tuple[int, int]],
    require_completion_effect: bool = False,
    require_shrink_effect: bool = False,
) -> list[dict]:
    h, w = grid_shape
    outline = _diamond_outline(center, radius)
    full_local = shape["full"]
    visible_local = shape["visible"]
    new_radius = radius - 1
    placements = []
    for r0 in range(h - shape["height"] + 1):
        for c0 in range(w - shape["width"] + 1):
            full_abs = _shift(full_local, r0, c0)
            visible_abs = _shift(visible_local, r0, c0)
            full_set = set(full_abs)
            visible_set = set(visible_abs)
            if full_set & outline:
                continue
            if full_set & occupied_full:
                continue
            if any(pt in blocked_visible for pt in visible_set):
                continue
            if any(pt in forbid for pt in full_set):
                continue
            cls = _classify(full_abs, center, new_radius)
            if cls != desired:
                continue
            visible_cls = _classify(visible_abs, center, new_radius)
            orig_cls = _classify(full_abs, center, radius)
            if require_completion_effect and visible_cls == cls:
                continue
            if require_shrink_effect and orig_cls == cls:
                continue
            if desired == "cross":
                cr, cc = center
                dists = [abs(r - cr) + abs(c - cc) for r, c in full_abs]
                if not (any(d < new_radius for d in dists) and any(d >= new_radius for d in dists)):
                    continue
            placements.append(
                {
                    "full": full_abs,
                    "visible": visible_abs,
                    "class_shrunk": cls,
                    "class_visible_shrunk": visible_cls,
                    "class_full_original": orig_cls,
                }
            )
    random.shuffle(placements)
    return placements


def _semantic_checks(inp: np.ndarray, placed: list[dict]) -> None:
    out, meta, center, radius = _solve_with_meta(inp)
    new_radius = radius - 1
    assert len(meta) == len(placed), "visible sprite count changed"
    by_visible = {tuple(item["visible"]): item for item in placed}
    completion_effect = False
    shrink_effect = False

    for item in meta:
        plan = by_visible.get(item["visible"])
        assert plan is not None, "solver changed sprite partition"
        assert item["full"] == plan["full"], "solver reconstructed the wrong completed sprite"
        assert item["class_shrunk"] == plan["class_shrunk"], "solver assigned the wrong class"
        completion_effect |= item["class_visible_shrunk"] != item["class_shrunk"]
        shrink_effect |= item["class_full_original"] != item["class_shrunk"]

    assert completion_effect, "generation must require mirror completion"
    assert shrink_effect, "generation must require diamond shrinking"

    counts = {cls: 0 for cls in ("inside", "outside", "cross")}
    for item in meta:
        counts[item["class_shrunk"]] += 1
    assert counts["inside"] >= 1 and counts["outside"] >= 1 and counts["cross"] >= 1

    cr, cc = center
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            d = abs(r - cr) + abs(c - cc)
            expected_red = d == new_radius
            if expected_red:
                assert out[r, c] in (RED, BLUE, GREEN)  # red can be replaced only by recolored inside/outside sprites, never happens here
    for item in meta:
        if item["class_shrunk"] == "inside":
            for r, c in item["full"]:
                assert out[r, c] == BLUE
        elif item["class_shrunk"] == "outside":
            for r, c in item["full"]:
                assert out[r, c] == GREEN
        else:
            for r, c in item["full"]:
                d = abs(r - cr) + abs(c - cc)
                assert out[r, c] == (RED if d == new_radius else BLACK)


def _attempt_generate() -> np.ndarray | None:
    h = random.randrange(*GRID_SIDE_RANGE)
    w = random.randrange(*GRID_SIDE_RANGE)
    radius = random.randrange(*RADIUS_RANGE)
    if min(h, w) < 2 * radius + 5:
        return None

    center = (
        random.randint(radius + 2, h - radius - 3),
        random.randint(radius + 2, w - radius - 3),
    )
    outline = _diamond_outline(center, radius)
    catalog = list(_shape_catalog())
    random.shuffle(catalog)

    sprite_count = random.randrange(*SPRITE_COUNT_RANGE)
    desired_roles = [
        ("cross", True, False),
        ("cross", False, True),
        ("inside", False, False),
        ("outside", False, False),
    ]
    while len(desired_roles) < sprite_count:
        cls = random.choice(["inside", "outside", "cross"])
        desired_roles.append((cls, False, False))

    placed = []
    occupied_full: set[tuple[int, int]] = set()
    blocked_visible: set[tuple[int, int]] = set(outline)
    forbid = set()

    for desired, need_completion, need_shrink in desired_roles:
        found = None
        for shape in catalog:
            placements = _candidate_placements(
                shape,
                (h, w),
                center,
                radius,
                desired,
                occupied_full,
                blocked_visible,
                forbid,
                require_completion_effect=need_completion,
                require_shrink_effect=need_shrink,
            )
            if placements:
                found = random.choice(placements)
                break
        if found is None:
            return None
        placed.append(found)
        occupied_full |= _sprite_neighborhood(found["full"])
        blocked_visible |= _sprite_neighborhood(found["visible"])
        forbid |= set(found["visible"])

    inp = np.full((h, w), BLACK, dtype=int)
    for r, c in outline:
        inp[r, c] = RED
    for item in placed:
        for r, c in item["visible"]:
            inp[r, c] = GREY

    if len(_components(inp == GREY)) != len(placed):
        return None

    try:
        _semantic_checks(inp, placed)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
