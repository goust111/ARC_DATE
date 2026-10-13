"""ARC puzzle 6: infer an unpainted light source and restore one missing sprite."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


RED = 2
GREY = 5
_DIRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
         if (dr, dc) != (0, 0)]


def _components(grid: np.ndarray, color: int) -> list[dict]:
    """Monochromatic 8-components, using the common toolbox's (row, col) convention."""
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for sr, sc in np.argwhere(grid == color):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        pts = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in _DIRS:
                rr, cc = r + dr, c + dc
                if (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
                        and not seen[rr, cc] and grid[rr, cc] == color):
                    seen[rr, cc] = True
                    q.append((rr, cc))
        r0 = min(r for r, _ in pts)
        c0 = min(c for _, c in pts)
        r1 = max(r for r, _ in pts)
        c1 = max(c for _, c in pts)
        norm = frozenset((r - r0, c - c0) for r, c in pts)
        result.append({"pts": frozenset(pts), "bbox": (r0, c0, r1, c1),
                       "shape": norm, "h": r1 - r0 + 1, "w": c1 - c0 + 1})
    return result


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _tight_step(h: int, w: int, direction: tuple[int, int]) -> tuple[int, int]:
    """Translation that puts equal bboxes immediately beside one another."""
    dr, dc = direction
    if dr == 0:
        return 0, dc * w
    if dc == 0:
        return dr * h, 0
    distance = min(h, w)
    return dr * distance, dc * distance


def _pair_direction(body: dict, shadow: dict) -> tuple[int, int] | None:
    if (body["h"], body["w"], body["shape"]) != (
            shadow["h"], shadow["w"], shadow["shape"]):
        return None
    br, bc = body["bbox"][:2]
    sr, sc = shadow["bbox"][:2]
    direction = (_sgn(sr - br), _sgn(sc - bc))
    if direction == (0, 0):
        return None
    return direction if (sr - br, sc - bc) == _tight_step(
        body["h"], body["w"], direction) else None


def _on_ray(light: tuple[int, int], center2: tuple[int, int],
            direction: tuple[int, int]) -> bool:
    """Whether a doubled-coordinate center is on an exact principal ray."""
    vr = center2[0] - 2 * light[0]
    vc = center2[1] - 2 * light[1]
    dr, dc = direction
    if dr == 0:
        return vr == 0 and vc * dc > 0
    if dc == 0:
        return vc == 0 and vr * dr > 0
    return vr * dr > 0 and vc * dc > 0 and abs(vr) == abs(vc)


def _center2(obj: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    return r0 + r1, c0 + c1


def _analyze(grid: np.ndarray):
    bodies = _components(grid, RED)
    shadows = _components(grid, GREY)
    assert len(bodies) >= 3, "at least three body-shadow examples are required"
    assert len(shadows) == len(bodies) + 1, "expected exactly one orphan shadow"

    matches = []
    used_shadows = set()
    for bi, body in enumerate(bodies):
        candidates = [(si, _pair_direction(body, shadow))
                      for si, shadow in enumerate(shadows)]
        candidates = [(si, d) for si, d in candidates if d is not None]
        assert len(candidates) == 1, "each red body must have one unique tight shadow"
        si, direction = candidates[0]
        assert si not in used_shadows, "two bodies cannot share a shadow"
        used_shadows.add(si)
        matches.append((bi, si, direction))

    orphan_ids = set(range(len(shadows))) - used_shadows
    assert len(orphan_ids) == 1, "the orphan shadow must be unique"
    orphan_id = orphan_ids.pop()

    lights = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if all(_on_ray((r, c), _center2(bodies[bi]), direction)
                   for bi, _si, direction in matches):
                lights.append((r, c))
    assert len(lights) == 1, "body-shadow direction rays must identify one light cell"
    light = lights[0]
    assert grid[light] not in (RED, GREY), "the light source must be invisible"
    return bodies, shadows, matches, orphan_id, light


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the invisible light source and paint the missing red body beside its orphan shadow.

    Input structure:
    - The dominant background contains at least three red connected sprites, each
      with an identically shaped grey copy tightly offset in one of eight directions.
    - Sprite shapes and sizes distinguish their pairings; exactly one additional
      grey sprite has no red counterpart, and the light itself is not painted.

    Rule:
    - Match equal red/grey sprites and reverse their body-to-shadow direction rays;
      their unique intersection is the invisible light source.
    - Translate the orphan grey sprite by one tight offset toward that source and
      paint the translated same-shape sprite red, preserving every input pixel.

    Key insight:
    - No fixed offset locates the missing body: the converging direction field of
      the complete pairs first reveals the reference point that reverses its shadow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _bodies, shadows, _matches, orphan_id, light = _analyze(g)
    orphan = shadows[orphan_id]
    cr2, cc2 = _center2(orphan)
    vr, vc = cr2 - 2 * light[0], cc2 - 2 * light[1]
    direction = (_sgn(vr), _sgn(vc))
    assert _on_ray(light, (cr2, cc2), direction), \
        "orphan shadow must lie on one of the light's eight principal rays"
    sr, sc = _tight_step(orphan["h"], orphan["w"], direction)

    out = g.copy()
    targets = {(r - sr, c - sc) for r, c in orphan["pts"]}
    assert all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in targets)
    bg = Counter(map(int, g.ravel())).most_common(1)[0][0]
    assert all(int(g[r, c]) == bg for r, c in targets), \
        "restored body cells must be empty background"
    for r, c in targets:
        out[r, c] = RED
    return out


_BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.ones((3, 3), dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.ones((3, 5), dtype=bool),
    np.array([[1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1]], dtype=bool),
    np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=bool),
]


def _shape_key(mask: np.ndarray):
    return mask.shape, mask.tobytes()


def _shape_pool() -> list[np.ndarray]:
    pool = {}
    for base in _BASE_SHAPES:
        for k in range(4):
            turned = np.rot90(base, k).copy()
            for mask in (turned, np.fliplr(turned).copy()):
                pool[_shape_key(mask)] = mask
    return list(pool.values())


def _pixels(mask: np.ndarray, top: tuple[int, int]) -> set[tuple[int, int]]:
    return {(top[0] + int(r), top[1] + int(c)) for r, c in np.argwhere(mask)}


def _far_from(groups: list[set[tuple[int, int]]], candidate: set[tuple[int, int]]) -> bool:
    halo = {(r + dr, c + dc) for r, c in candidate
            for dr in range(-2, 3) for dc in range(-2, 3)}
    return all(not (halo & group) for group in groups)


def _generate_attempt() -> np.ndarray | None:
    h, w = random.randint(22, 30), random.randint(22, 30)
    light = (random.randint(7, h - 8), random.randint(7, w - 8))
    pair_count = random.randint(3, 5)
    count = pair_count + 1

    # Every example mixes axial and diagonal branches and has intersecting line classes.
    for _ in range(30):
        directions = random.sample(_DIRS, count)
        classes = {(0 if dr == 0 else 1 if dc == 0 else 2 if dr == dc else 3)
                   for dr, dc in directions[:pair_count]}
        if len(classes) >= 2 and any(dr == 0 or dc == 0 for dr, dc in directions) \
                and any(dr != 0 and dc != 0 for dr, dc in directions):
            break
    else:
        return None

    shapes = random.sample(_shape_pool(), count)
    groups = []
    records = []
    for mask, direction in zip(shapes, directions):
        mh, mw = mask.shape
        placed = None
        for distance in random.sample(range(4, 10), 6):
            center = (light[0] + direction[0] * distance,
                      light[1] + direction[1] * distance)
            top = (center[0] - mh // 2, center[1] - mw // 2)
            step = _tight_step(mh, mw, direction)
            shadow_top = (top[0] + step[0], top[1] + step[1])
            body_pts = _pixels(mask, top)
            shadow_pts = _pixels(mask, shadow_top)
            whole = body_pts | shadow_pts
            if (all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in whole)
                    and light not in whole and _far_from(groups, whole)):
                placed = (body_pts, shadow_pts, whole)
                break
        if placed is None:
            return None
        body_pts, shadow_pts, whole = placed
        groups.append(whole)
        records.append((body_pts, shadow_pts))

    bg = random.choice([0, 1, 3, 4, 6, 7, 8, 9])
    grid = np.full((h, w), bg, dtype=int)
    for i, (body_pts, shadow_pts) in enumerate(records):
        if i != pair_count:  # the final record contributes only its orphan shadow
            for r, c in body_pts:
                grid[r, c] = RED
        for r, c in shadow_pts:
            grid[r, c] = GREY

    try:
        _b, _s, _m, orphan_id, inferred = _analyze(grid)
        if inferred != light:
            return None
        expected = grid.copy()
        for r, c in records[-1][0]:
            expected[r, c] = RED
        if not np.array_equal(main(grid), expected):
            return None
        # Non-degeneracy: the orphan is not always on the same side as any known pair.
        orphan = _s[orphan_id]
        if len(orphan["pts"]) == 0:
            return None
    except (AssertionError, IndexError):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, semantically validated instance of the invisible-light task."""
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-light input")
