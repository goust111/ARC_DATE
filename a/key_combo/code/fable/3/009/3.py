import itertools
import random
from collections import Counter

import numpy as np


BG = 0
NON_BG = list(range(1, 10))
GEN_ATTEMPTS = 500


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        pts = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(pts, dtype=int))
    return out


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for coords in _components(grid != bg):
        comps.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
                "colors": set(map(int, grid[coords[:, 0], coords[:, 1]])),
            }
        )
    return comps


def _is_hollow_rectangle(comp: dict) -> bool:
    if len(comp["colors"]) != 1:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    if comp["area"] != 2 * h + 2 * w - 4:
        return False
    pts = {(int(r), int(c)) for r, c in comp["coords"]}
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    return all(r in (r0, r1) or c in (c0, c1) for r, c in pts)


def _inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _scene(grid: np.ndarray) -> tuple[list[dict], int]:
    bg = _background(grid)
    comps = _foreground_components(grid, bg)
    frames = [comp for comp in comps if _is_hollow_rectangle(comp)]
    assert len(frames) == 1, "expected one rectangular frame"
    frame = frames[0]

    framed = [comp for comp in comps if comp is not frame and _inside(comp["bbox"], frame["bbox"])]
    assert len(framed) == 1, "expected one sprite inside the frame"
    framed_box = framed[0]["bbox"]
    framed_shape = _crop(grid, framed_box).shape
    assert framed_shape[0] == framed_shape[1], "rotated copies must have square bboxes"

    sprites = []
    for comp in comps:
        if comp is frame:
            continue
        patch = _crop(grid, comp["bbox"])
        if patch.shape == framed_shape:
            sprites.append(
                {
                    "bbox": comp["bbox"],
                    "patch": patch,
                    "framed": comp["bbox"] == framed_box,
                }
            )
    assert sum(1 for sp in sprites if sp["framed"]) == 1
    assert len(sprites) >= 4
    return sprites, next(i for i, sp in enumerate(sprites) if sp["framed"])


def _majority(aligned: list[np.ndarray]) -> np.ndarray | None:
    h, w = aligned[0].shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            counts = Counter(int(tile[r, c]) for tile in aligned)
            color, votes = counts.most_common(1)[0]
            if votes < 3:
                return None
            out[r, c] = int(color)
    return out


def _candidate_solutions(grid: np.ndarray) -> list[np.ndarray]:
    sprites, framed_idx = _scene(grid)
    other_indices = [i for i in range(len(sprites)) if i != framed_idx]
    framed = sprites[framed_idx]["patch"]
    solutions = []

    for chosen in itertools.combinations(other_indices, 3):
        for turns in itertools.permutations((1, 2, 3)):
            aligned = [framed]
            for idx, k in zip(chosen, turns):
                aligned.append(np.rot90(sprites[idx]["patch"], k))
            maj = _majority(aligned)
            if maj is not None:
                solutions.append(maj)

    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the framed orientation of four noisy rotated sprite copies by majority vote.

    Input structure:
    - A dominant background contains five equal-size square multicolor sprites that are separated from each other.
    - Four sprites are the 0, 90, 180, and 270 degree rotations of one source tile, each with a few wrong-color defect pixels.
    - One same-size sprite is an impostor, and exactly one true copy is enclosed by a single-color hollow rectangular frame.

    Rule:
    - Detect the hollow frame and use the enclosed sprite as the output orientation anchor.
    - Among the equal-size sprites, choose the unique set of four that can be rotated by distinct quarter-turns into the framed orientation so every cell has at least three matching votes.
    - Output the per-cell majority colors of those four aligned sprites, excluding the frame.

    Key insight:
    - Exact equality fails because every true copy is noisy; the impostor is excluded because it breaks the three-of-four rotational voting consistency.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    solutions = _candidate_solutions(g)
    unique = []
    for sol in solutions:
        if not any(np.array_equal(sol, prev) for prev in unique):
            unique.append(sol)
    assert len(unique) == 1, "expected a unique repaired tile"
    return unique[0]


def _rotate_coord(r: int, c: int, n: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return n - 1 - c, r
    if k == 2:
        return n - 1 - r, n - 1 - c
    return c, n - 1 - r


def _make_source(n: int, palette: list[int]) -> np.ndarray:
    for _ in range(200):
        tile = np.array([[random.choice(palette) for _ in range(n)] for _ in range(n)], dtype=int)
        if len(set(map(int, tile.flatten()))) < min(4, len(palette)):
            continue
        min_diff = min(int(np.count_nonzero(tile != np.rot90(tile, k))) for k in (1, 2, 3))
        if min_diff >= max(8, (n * n * 3) // 5):
            return tile
    raise RuntimeError("failed to make asymmetric source tile")


def _make_impostor(source: np.ndarray, palette: list[int]) -> np.ndarray | None:
    n = int(source.shape[0])
    threshold = max(8, (n * n * 3) // 5)
    colors = list(dict.fromkeys(int(c) for c in source.flatten()))

    for _ in range(120):
        shuffled = colors[:]
        random.shuffle(shuffled)
        if any(a == b for a, b in zip(colors, shuffled)):
            continue
        mapping = dict(zip(colors, shuffled))
        imp = np.vectorize(mapping.get)(source).astype(int)
        imp = np.rot90(imp, random.randrange(4)).copy()
        if all(int(np.count_nonzero(imp != np.rot90(source, k))) >= threshold for k in range(4)):
            return imp

    for _ in range(200):
        imp = np.array([[random.choice(palette) for _ in range(n)] for _ in range(n)], dtype=int)
        if all(int(np.count_nonzero(imp != np.rot90(source, k))) >= threshold for k in range(4)):
            return imp
    return None


def _add_defects(copies: list[np.ndarray], frame_color: int) -> None:
    n = int(copies[0].shape[0])
    coords = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(coords)
    max_each = max(2, n * n // 9)
    counts = [random.randint(1, max_each) for _ in range(4)]
    assert sum(counts) <= len(coords)

    cursor = 0
    defect_palette = [c for c in NON_BG if c != frame_color]
    copy_order = list(range(4))
    random.shuffle(copy_order)
    for idx in copy_order:
        for r, c in coords[cursor : cursor + counts[idx]]:
            rr, cc = _rotate_coord(r, c, n, idx)
            old = int(copies[idx][rr, cc])
            choices = [c for c in defect_palette if c != old]
            copies[idx][rr, cc] = random.choice(choices)
        cursor += counts[idx]


def _framed_patch(sprite: np.ndarray, frame_color: int) -> np.ndarray:
    n = int(sprite.shape[0])
    patch = np.full((n + 4, n + 4), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[2 : 2 + n, 2 : 2 + n] = sprite
    return patch


def _dilated(mask: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        out[r0:r1, c0:c1] = True
    return out


def _place_random(grid: np.ndarray, patch: np.ndarray, *, border: int = 1, padding: int = 1) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    mask = patch != BG
    blocked = _dilated(grid != BG, padding)
    choices = []
    for r0 in range(border, gh - h - border + 1):
        for c0 in range(border, gw - w - border + 1):
            if not bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                choices.append((r0, c0))
    if not choices:
        return False
    r0, c0 = random.choice(choices)
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]
    return True


def _make_attempt(frame_turn: int | None = None) -> np.ndarray | None:
    n = random.choice([4, 5, 6])
    palette_size = random.choice([4, 5])
    palette = random.sample(NON_BG, palette_size)
    source = _make_source(n, palette)
    frame_choices = [c for c in NON_BG if c not in palette]
    if not frame_choices:
        return None
    frame_color = random.choice(frame_choices)

    copies = [np.rot90(source, k).copy() for k in range(4)]
    _add_defects(copies, frame_color)
    impostor = _make_impostor(source, palette)
    if impostor is None:
        return None

    framed_idx = random.randrange(4) if frame_turn is None else int(frame_turn) % 4
    expected = np.rot90(source, framed_idx).copy()

    patches = []
    for idx, sprite in enumerate(copies):
        patches.append(_framed_patch(sprite, frame_color) if idx == framed_idx else sprite)
    patches.append(impostor)
    random.shuffle(patches)
    patches.sort(key=lambda p: int(p.size), reverse=True)

    gh = random.randint(max(24, n + 16), 30)
    gw = random.randint(max(24, n + 16), 30)
    grid = np.full((gh, gw), BG, dtype=int)
    for patch in patches:
        if not _place_random(grid, patch, border=1, padding=1):
            return None

    try:
        solved = main(grid)
        if not np.array_equal(solved, expected):
            return None
        if len(_candidate_solutions(grid)) != 1:
            return None
    except Exception:
        return None
    return grid


_GEN_CALLS = 0


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    preferred_turn = _GEN_CALLS % 4
    _GEN_CALLS += 1
    for attempt in range(GEN_ATTEMPTS):
        frame_turn = preferred_turn if attempt < GEN_ATTEMPTS // 2 else None
        grid = _make_attempt(frame_turn=frame_turn)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid rotated-vote repair input")
