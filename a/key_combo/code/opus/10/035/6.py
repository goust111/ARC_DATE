from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
OCCLUDER = 9
PATCH_COLORS = (BG, RED, YELLOW)

SIDE_OPTIONS_SMALL = (15, 17, 19, 21, 23)
SIDE_OPTIONS_LARGE = (19, 21, 23)
SQUARE_SIZE_OPTIONS = (3, 4)
GEN_ATTEMPTS = 400
MOTIF_ATTEMPTS = 60
PATTERN_ATTEMPTS = 120
POSITION_ATTEMPTS = 120

MOTIFS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _square_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            bbox = _bbox_from_coords(coords)
            side_r = bbox[2] - bbox[0] + 1
            side_c = bbox[3] - bbox[1] + 1
            assert side_r == side_c, "expected square occluders"
            assert len(coords) == side_r * side_c, "occluders must be solid squares"
            comps.append(
                {
                    "bbox": bbox,
                    "size": side_r,
                    "coords": coords,
                }
            )

    return sorted(comps, key=lambda comp: comp["bbox"])


def _mirror_bbox_lr(bbox: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0, width - 1 - c1, r1, width - 1 - c0


def _patch_key(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in patch.tolist())


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Recover all mirror-hidden square patches and output the one recovered pattern that appears only once.

    Input structure
    - The grid has black background with a left-right mirror-symmetric red/yellow scene.
    - Three maroon solid square occluders of the same size sit on the left side of the vertical midline.
    - The mirrored locations on the right side are unobstructed and show the visible counterparts of the hidden patches.

    Rule
    - Detect the three maroon solid squares and record each square bbox.
    - Reflect every bbox across the vertical midline, extract the mirrored patch, and flip that patch horizontally to reconstruct the hidden content.
    - Compare the three reconstructed patches pixel-by-pixel and return the single patch whose content is not duplicated.

    Key insight
    - The candidates to compare do not exist visibly on the left at all; they must be reconstructed from mirror symmetry before the odd-one-out comparison can be made.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    occluders = _square_components(g, OCCLUDER)
    assert len(occluders) == 3, "expected exactly three occluder squares"

    side_lengths = {int(comp["size"]) for comp in occluders}
    assert len(side_lengths) == 1, "occluders must all share the same square size"
    side = next(iter(side_lengths))
    axis = w // 2

    for a, b in combinations(occluders, 2):
        assert _boxes_separated(a["bbox"], b["bbox"], gap=1), "occluders must not touch each other"

    recovered: list[np.ndarray] = []
    for comp in occluders:
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        assert c1 < axis, "all occluders must lie left of the vertical midline"
        assert c1 <= axis - 2, "occluders must not touch the vertical midline"
        mb = _mirror_bbox_lr(bbox, w)
        mr0, mc0, mr1, mc1 = mb
        assert 0 <= mr0 <= mr1 < h and 0 <= mc0 <= mc1 < w
        patch = g[mr0 : mr1 + 1, mc0 : mc1 + 1]
        assert patch.shape == (side, side), "mirrored patch shape mismatch"
        assert not np.any(patch == OCCLUDER), "mirrored source must be unobstructed"
        recovered.append(np.fliplr(patch))

    keys = [_patch_key(patch) for patch in recovered]
    counts = Counter(keys)
    unique_keys = [key for key, count in counts.items() if count == 1]
    assert len(unique_keys) == 1, "expected exactly one reconstructed patch to be unique"

    unique_idx = keys.index(unique_keys[0])
    return recovered[unique_idx].copy()


def _valid_patch(patch: np.ndarray) -> bool:
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        return False
    if set(map(int, np.unique(patch))) - set(PATCH_COLORS):
        return False
    if np.count_nonzero(patch != BG) < patch.shape[0] + 1:
        return False
    if not np.any(patch == RED) or not np.any(patch == YELLOW):
        return False
    if np.array_equal(patch, np.fliplr(patch)):
        return False
    return True


def _random_patch(side: int) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        flat = random.choices(PATCH_COLORS, weights=(5, 4, 4), k=side * side)
        patch = np.array(flat, dtype=int).reshape(side, side)
        if _valid_patch(patch):
            return patch
    raise RuntimeError("failed to generate a valid patch")


def _mutate_patch(base: np.ndarray) -> np.ndarray:
    side = base.shape[0]
    for _ in range(PATTERN_ATTEMPTS):
        cand = base.copy()
        change_count = random.randint(2, min(side + 1, side * side))
        for idx in random.sample(range(side * side), change_count):
            r, c = divmod(idx, side)
            options = [color for color in PATCH_COLORS if color != int(cand[r, c])]
            cand[r, c] = random.choice(options)
        if _valid_patch(cand) and not np.array_equal(cand, base):
            if not np.array_equal(cand, np.fliplr(base)):
                return cand
    raise RuntimeError("failed to mutate patch")


def _candidate_positions(side: int, sq: int) -> list[tuple[int, int]]:
    axis = side // 2
    rows = range(0, side - sq + 1)
    cols = range(0, axis - sq)
    return [(r, c) for r in rows for c in cols]


def _choose_positions(side: int, sq: int) -> list[tuple[int, int]]:
    candidates = _candidate_positions(side, sq)
    if len(candidates) < 3:
        raise RuntimeError("not enough candidate occluder positions")

    for _ in range(POSITION_ATTEMPTS):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        boxes: list[tuple[int, int, int, int]] = []
        for r0, c0 in candidates:
            box = (r0, c0, r0 + sq - 1, c0 + sq - 1)
            if all(_boxes_separated(box, other, gap=1) for other in boxes):
                chosen.append((r0, c0))
                boxes.append(box)
                if len(chosen) == 3:
                    return chosen
    raise RuntimeError("failed to place three separated occluders")


def _paint_lr(scene: np.ndarray, r: int, c: int, color: int) -> None:
    h, w = scene.shape
    assert 0 <= r < h and 0 <= c < w
    scene[r, c] = int(color)
    scene[r, w - 1 - c] = int(color)


def _place_patch_pair(scene: np.ndarray, r0: int, c0: int, patch: np.ndarray) -> None:
    side = patch.shape[0]
    scene[r0 : r0 + side, c0 : c0 + side] = patch
    mirror_c0 = scene.shape[1] - c0 - side
    scene[r0 : r0 + side, mirror_c0 : mirror_c0 + side] = np.fliplr(patch)


def _reserve_exact_cells(side: int, positions: list[tuple[int, int]], sq: int) -> np.ndarray:
    reserved = np.zeros((side, side), dtype=bool)
    for r0, c0 in positions:
        reserved[r0 : r0 + sq, c0 : c0 + sq] = True
        mc0 = side - c0 - sq
        reserved[r0 : r0 + sq, mc0 : mc0 + sq] = True
    return reserved


def _try_place_motif(scene: np.ndarray, reserved: np.ndarray, motif: tuple[tuple[int, int], ...], color: int) -> bool:
    side = scene.shape[0]
    max_r = max(r for r, _ in motif)
    max_c = max(c for _, c in motif)
    axis = side // 2
    if axis - max_c <= 0:
        return False

    for _ in range(MOTIF_ATTEMPTS):
        r0 = random.randint(0, side - max_r - 1)
        c0 = random.randint(0, axis - max_c - 1)
        cells = [(r0 + dr, c0 + dc) for dr, dc in motif]
        mirrors = [(r, side - 1 - c) for r, c in cells]
        all_cells = cells + mirrors
        if any(reserved[r, c] for r, c in all_cells):
            continue
        if any(scene[r, c] != BG for r, c in all_cells):
            continue
        for r, c in cells:
            _paint_lr(scene, r, c, color)
        return True
    return False


def _add_axis_details(scene: np.ndarray, reserved: np.ndarray) -> None:
    side = scene.shape[0]
    axis = side // 2
    for _ in range(random.randint(1, 4)):
        length = random.randint(1, 3)
        r0 = random.randint(0, side - length)
        if np.any(reserved[r0 : r0 + length, axis]) or np.any(scene[r0 : r0 + length, axis] != BG):
            continue
        scene[r0 : r0 + length, axis] = random.choice((RED, YELLOW))


def _build_scene(side: int, sq: int, positions: list[tuple[int, int]], patches: list[np.ndarray]) -> np.ndarray:
    scene = np.full((side, side), BG, dtype=int)
    for (r0, c0), patch in zip(positions, patches):
        _place_patch_pair(scene, r0, c0, patch)

    reserved = _reserve_exact_cells(side, positions, sq)
    motif_target = random.randint(side // 2 + 4, side + 6)
    for _ in range(motif_target):
        _try_place_motif(scene, reserved, random.choice(MOTIFS), random.choice((RED, YELLOW)))

    _add_axis_details(scene, reserved)
    return scene


def _semantic_check(inp: np.ndarray, clean: np.ndarray, expected: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False
    if not np.array_equal(clean, np.fliplr(clean)):
        return False

    occluders = _square_components(inp, OCCLUDER)
    if len(occluders) != 3:
        return False

    recovered = []
    for comp in occluders:
        r0, c0, r1, c1 = comp["bbox"]
        mb = _mirror_bbox_lr(comp["bbox"], inp.shape[1])
        mr0, mc0, mr1, mc1 = mb
        visible = inp[mr0 : mr1 + 1, mc0 : mc1 + 1]
        hidden = clean[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(visible == OCCLUDER):
            return False
        if not np.array_equal(np.fliplr(visible), hidden):
            return False
        recovered.append(hidden.copy())

    counts = Counter(_patch_key(patch) for patch in recovered)
    if sorted(counts.values()) != [1, 2]:
        return False
    return True


def _generate_once() -> np.ndarray:
    sq = random.choice(SQUARE_SIZE_OPTIONS)
    side = random.choice(SIDE_OPTIONS_SMALL if sq == 3 else SIDE_OPTIONS_LARGE)
    positions = _choose_positions(side, sq)

    repeated = _random_patch(sq)
    unique = _mutate_patch(repeated)
    unique_idx = random.randrange(3)
    patches = [repeated.copy(), repeated.copy(), repeated.copy()]
    patches[unique_idx] = unique

    clean = _build_scene(side, sq, positions, patches)
    inp = clean.copy()
    for r0, c0 in positions:
        inp[r0 : r0 + sq, c0 : c0 + sq] = OCCLUDER

    expected = unique.copy()
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert expected.ndim == 2 and max(expected.shape) <= 30
    if not _semantic_check(inp, clean, expected):
        raise RuntimeError("semantic check failed")
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _generate_once()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid mirrored-occluder uniqueness puzzle")
