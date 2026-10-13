import random
from collections import Counter

import numpy as np


BG = 0
OBJ_COLOR = 5
KINDS = ("H", "V", "R")
SAMPLE_MASKS = {
    "H": np.array([[1, 0], [1, 0]], dtype=bool),
    "V": np.array([[1, 1], [0, 0]], dtype=bool),
    "R": np.array([[1, 0], [0, 1]], dtype=bool),
}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _classify(mask: np.ndarray) -> str | None:
    h = bool(np.array_equal(mask, np.flipud(mask)))
    v = bool(np.array_equal(mask, np.fliplr(mask)))
    r = bool(np.array_equal(mask, np.rot90(mask, 2)))
    if (h + v + r) != 1:
        return None
    if h:
        return "H"
    if v:
        return "V"
    return "R"


def _legend_entries(grid: np.ndarray, bg: int) -> dict[str, int]:
    h, w = grid.shape
    entries: dict[str, int] = {}
    for r in range(h - 1):
        patch = grid[r : r + 2, : min(3, w)]
        if patch.shape != (2, 3):
            continue
        if not (patch[:, :2] != bg).any():
            continue
        if r > 0 and bool((grid[r - 1, :3] != bg).any()):
            continue
        if r + 2 < h and bool((grid[r + 2, :3] != bg).any()):
            pass
        sample = patch[:, :2] != bg
        if sample.sum() == 0:
            continue
        color_pts = np.argwhere(patch[:, 2] != bg)
        if len(color_pts) != 1:
            continue
        kind = _classify(sample)
        if kind is None:
            continue
        entries[kind] = int(patch[color_pts[0][0], 2])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    legend = _legend_entries(g, bg)
    assert set(legend) == set(KINDS)

    obj_mask = g == OBJ_COLOR
    comps = _components(obj_mask)
    assert comps

    out = np.full_like(g, bg)
    for coords in comps:
        mask, (r0, c0, r1, c1) = _tight_mask(coords)
        kind = _classify(mask)
        assert kind is not None
        ph, pw = mask.shape
        color = legend[kind]

        if kind == "H":
            assert (h - ph) % 2 == 0
            nr0, nc0 = (h - ph) // 2, c0
        elif kind == "V":
            assert (w - pw) % 2 == 0
            nr0, nc0 = r0, (w - pw) // 2
        else:
            assert (h - ph) % 2 == 0 and (w - pw) % 2 == 0
            nr0, nc0 = (h - ph) // 2, (w - pw) // 2

        target = out[nr0 : nr0 + ph, nc0 : nc0 + pw]
        assert not bool((target != bg)[mask].any())
        target[mask] = color
    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    stack = [tuple(map(int, pts[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _orbits(kind: str, h: int, w: int) -> list[list[tuple[int, int]]]:
    seen = set()
    out = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            if kind == "H":
                orb = {(r, c), (h - 1 - r, c)}
            elif kind == "V":
                orb = {(r, c), (r, w - 1 - c)}
            else:
                orb = {(r, c), (h - 1 - r, w - 1 - c)}
            orb = sorted(orb)
            seen.update(orb)
            out.append(orb)
    return out


def _grow_mask(kind: str, h: int, w: int) -> np.ndarray | None:
    orbits = _orbits(kind, h, w)
    fixed = []
    for orb in orbits:
        if len(orb) == 1:
            fixed.append(orb)
    if kind == "R":
        seed = fixed[0] if fixed else random.choice(orbits)
    else:
        axis_orbits = [orb for orb in fixed if (orb[0][0] == h // 2 if kind == "H" else orb[0][1] == w // 2)]
        seed = random.choice(axis_orbits if axis_orbits else fixed if fixed else orbits)

    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        for r, c in seed:
            mask[r, c] = True
        others = [orb for orb in orbits if orb != seed]
        random.shuffle(others)
        target = random.randint(max(5, (h + w) // 2), min(h * w - 1, max(7, (3 * h * w) // 5)))
        for orb in others:
            if int(mask.sum()) >= target:
                break
            tmp = mask.copy()
            for r, c in orb:
                tmp[r, c] = True
            if _is_connected(tmp):
                mask = tmp
                if random.random() < 0.25 and int(mask.sum()) >= 5:
                    break
        if int(mask.sum()) >= 5 and _is_connected(mask) and _classify(mask) == kind:
            return mask
    return None


def _paint_entry(grid: np.ndarray, top: int, kind: str, color: int) -> None:
    sample = SAMPLE_MASKS[kind]
    grid[top : top + 2, :2][sample] = color
    grid[top + random.randrange(2), 2] = color


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    sub = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    sub[mask] = color


def _sample_object(kind: str, gh: int, gw: int) -> np.ndarray:
    for _ in range(100):
        h = random.randrange(3, min(10, gh - 3), 2)
        w = random.randrange(3, min(10, gw - 5), 2)
        mask = _grow_mask(kind, h, w)
        if mask is not None:
            return mask
    raise RuntimeError(f"failed to build {kind} object")


def _target_box(kind: str, mask: np.ndarray, pos: tuple[int, int], gh: int, gw: int) -> tuple[int, int, int, int]:
    r0, c0 = pos
    h, w = mask.shape
    if kind == "H":
        r0 = (gh - h) // 2
    elif kind == "V":
        c0 = (gw - w) // 2
    else:
        r0 = (gh - h) // 2
        c0 = (gw - w) // 2
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _gen_attempt() -> np.ndarray | None:
    gh = random.randrange(17, 26, 2)
    gw = random.randrange(17, 28, 2)
    grid = np.full((gh, gw), BG, dtype=int)

    legend_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    legend_order = random.sample(list(KINDS), 3)
    for top, kind in zip((1, 4, 7), legend_order):
        _paint_entry(grid, top, kind, legend_colors[KINDS.index(kind)])

    kinds = ["H", "V", "R"]
    for _ in range(random.randint(0, 2)):
        kinds.append(random.choice(["H", "V"]))
    random.shuffle(kinds)

    masks = []
    for kind in kinds:
        masks.append((kind, _sample_object(kind, gh, gw)))

    placed: list[tuple[str, np.ndarray, tuple[int, int]]] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    target_boxes: list[tuple[int, int, int, int]] = []

    for kind, mask in masks:
        mh, mw = mask.shape
        ok = False
        for _ in range(200):
            r0 = random.randint(0, gh - mh)
            c0 = random.randint(5, gw - mw)
            box = (r0, c0, r0 + mh - 1, c0 + mw - 1)
            if any(_rects_overlap(box, prev, pad=1) for prev in input_boxes):
                continue
            if c0 <= 3:
                continue
            tbox = _target_box(kind, mask, (r0, c0), gh, gw)
            if any(_rects_overlap(tbox, prev) for prev in target_boxes):
                continue
            placed.append((kind, mask, (r0, c0)))
            input_boxes.append(box)
            target_boxes.append(tbox)
            ok = True
            break
        if not ok:
            return None

    if all(_target_box(kind, mask, pos, gh, gw) == (pos[0], pos[1], pos[0] + mask.shape[0] - 1, pos[1] + mask.shape[1] - 1) for kind, mask, pos in placed):
        return None

    for _, mask, (r0, c0) in placed:
        _paste_mask(grid, mask, r0, c0, OBJ_COLOR)

    counts = Counter(kind for kind, _, _ in placed)
    if not all(counts[k] >= 1 for k in KINDS):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
