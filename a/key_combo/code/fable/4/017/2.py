from __future__ import annotations

from pathlib import Path
import itertools
import sys

import numpy as np


_ROOT = Path(__file__).resolve()
for _parent in _ROOT.parents:
    _solver_root = _parent / "new" / "solvers"
    if (_solver_root / "arc_utils.py").exists():
        _solver_dir = str(_solver_root)
        if _solver_dir not in sys.path:
            sys.path.insert(0, _solver_dir)
        break

from arc_utils import Color, bbox, colors_present, crop_bbox, find_components, paste, random_sprite


def _bbox_includes_yellow_only(grid: np.ndarray, bbox_corner: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_corner)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return patch == Color.YELLOW


def _find_corner_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    """Return (r0,c0,r1,c1) for the four yellow corners."""
    corners = np.argwhere(grid == Color.YELLOW)
    assert corners.shape[0] == 4, "expected exactly 4 yellow corner pixels"
    r0 = int(corners[:, 0].min())
    c0 = int(corners[:, 1].min())
    r1 = int(corners[:, 0].max())
    c1 = int(corners[:, 1].max())
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    actual = {(int(r), int(c)) for r, c in corners.tolist()}
    assert actual == expected, "yellow pixels must be the four bounding box corners"
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h % 2 == 0 and h > 1, "expected a 2n x 2n yellow-marked square"
    return r0, c0, r1, c1


def _bbox_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    a0, b0, a1, b1 = map(int, a)
    c0, d0, c1, d1 = map(int, b)
    return a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0


def _extract_sprite(grid: np.ndarray, bg: int, target_box: tuple[int, int, int, int]) -> np.ndarray:
    objects = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="objects")
    candidates: list[np.ndarray] = []
    for obj in objects:
        b = bbox(obj != bg)
        if b is None:
            continue
        if not _bbox_disjoint(b, target_box):
            continue
        vals = np.unique(obj[obj != bg])
        if vals.size == 1 and vals[0] == Color.YELLOW:
            continue
        candidates.append(crop_bbox(obj, b))
    assert len(candidates) == 1, "expected exactly one sprite object outside the target box"
    sprite = candidates[0]
    assert sprite.ndim == 2
    return sprite


def _rotations(sprite: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(sprite, k) for k in range(4)]


def _quadrants(n: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, n), slice(0, n)),  # upper-left
        (slice(0, n), slice(n, 2 * n)),  # upper-right
        (slice(n, 2 * n), slice(0, n)),  # lower-left
        (slice(n, 2 * n), slice(n, 2 * n)),  # lower-right
    ]


def _find_assignments(frame: np.ndarray, sprite: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """Return all valid assignments (r0_rotation, r1_rotation, r2_rotation, r3_rotation)."""
    assert frame.shape[0] == frame.shape[1], "frame must be square"
    h = frame.shape[0]
    assert h % 2 == 0 and h > 1
    n = h // 2
    rots = _rotations(sprite)
    assert all(rot.shape == (n, n) for rot in rots)

    marker_mask = frame == Color.YELLOW
    quadrants = _quadrants(n)

    candidates: list[list[int]] = []
    for q_r, q_c in quadrants:
        q = frame[q_r, q_c]
        clue_mask = (q != bg) & (~marker_mask[q_r, q_c])
        clues = np.argwhere(clue_mask)
        valid: list[int] = []
        for idx, rot in enumerate(rots):
            ok = True
            for rr, cc in clues:
                if int(rot[int(rr), int(cc)]) != int(q[int(rr), int(cc)]):
                    ok = False
                    break
            if ok:
                valid.append(idx)
        candidates.append(valid)
        assert valid, "no rotation can satisfy this quadrant"

    assignments: list[tuple[int, int, int, int]] = []
    for perm in itertools.permutations(range(4)):
        if all(int(perm[i]) in candidates[i] for i in range(4)):
            assignments.append(tuple(map(int, perm)))
    return assignments


def _assemble_output(frame: np.ndarray, assignment: tuple[int, int, int, int], rots: list[np.ndarray]) -> np.ndarray:
    n2 = frame.shape[0] // 2
    out = frame.copy()
    marker_mask = frame == Color.YELLOW
    for qi, rot_idx in enumerate(assignment):
        q_r, q_c = _quadrants(n2)[qi]
        out[q_r, q_c] = rots[rot_idx]
    out[marker_mask] = Color.YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a yellow-cornered 2n×2n target by assigning one rotation of a full sprite to each quadrant.

    Input structure:
    - The input contains one full n×n multicolor sprite outside the marked region.
    - Four yellow singleton pixels mark the four corners of a 2n×2n target canvas.
    - Each quadrant of the target canvas contains sparse colored traces matching one rotation of the sprite.

    Rule:
    - Crop the 2n×2n area delimited by the yellow corners.
    - Generate the four 90° rotations of the external sprite and test each quadrant against its visible traces.
    - Enforce the global constraint that the four quadrants must use the four rotations exactly once, then choose the unique valid assignment.
    - Paste the selected rotations into their quadrants to produce the output canvas.

    Key insight:
    - Local traces may be too sparse to decide a quadrant orientation alone, so the full permutation constraint across all quadrants is required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(colors_present(g, exclude={Color.YELLOW})[0]) if np.any(g != Color.YELLOW) else int(Color.BLACK)
    corner_box = _find_corner_box(g)
    frame = g[corner_box[0] : corner_box[2] + 1, corner_box[1] : corner_box[3] + 1].copy()

    sprite = _extract_sprite(g, bg=bg, target_box=corner_box)
    n = frame.shape[0] // 2
    assert sprite.shape == (n, n), f"sprite must be {n}x{n}, got {sprite.shape}"

    assignments = _find_assignments(frame, sprite, bg)
    assert len(assignments) == 1, "rotation assignment must be unique"
    assignment = assignments[0]

    out = _assemble_output(frame, assignment, _rotations(sprite))
    out[_bbox_includes_yellow_only(g, corner_box)] = Color.YELLOW
    return out


def _draw_evidence(base: np.ndarray, keep_mode: str, rng: np.random.Generator, n: int) -> np.ndarray:
    frame = base.copy()
    marker_coords = np.argwhere(frame == Color.YELLOW)
    marker_set = {tuple(map(int, item)) for item in marker_coords}
    keep_mask = np.ones(frame.shape, dtype=bool)
    for r, c in marker_set:
        keep_mask[r, c] = False

    if keep_mode == "dense":
        keep = rng.random(frame.shape) < 0.88
        keep[keep_mask] = keep[keep_mask]
        keep[~keep_mask] = True
        frame[~keep] = Color.BLACK
    elif keep_mode == "medium":
        keep = rng.random(frame.shape) < 0.35
        keep[keep_mask] = keep[keep_mask]
        keep[~keep_mask] = True
        frame[~keep] = Color.BLACK
    else:  # sparse
        frame[:] = Color.BLACK
        frame[frame == Color.YELLOW] = Color.YELLOW
        quad = _quadrants(n)
        for q_r, q_c in quad:
            cells = [
                (int(r), int(c))
                for r in range(q_r.start, q_r.stop)
                for c in range(q_c.start, q_c.stop)
                if (int(r), int(c)) not in marker_set
            ]
            k = rng.integers(1, max(2, len(cells)))
            pick = rng.choice(len(cells), size=int(k), replace=False)
            for idx in pick:
                rr, cc = cells[int(idx)]
                frame[rr, cc] = base[rr, cc]
    return frame


def _assemble_full_frame(sprite: np.ndarray, assignment: tuple[int, int, int, int]) -> np.ndarray:
    n = sprite.shape[0]
    rots = _rotations(sprite)
    out = np.full((2 * n, 2 * n), Color.BLACK, dtype=int)
    for qi, ridx in enumerate(assignment):
        r_slice, c_slice = _quadrants(n)[qi]
        out[r_slice, c_slice] = rots[int(ridx)]
    out[0, 0] = Color.YELLOW
    out[0, 2 * n - 1] = Color.YELLOW
    out[2 * n - 1, 0] = Color.YELLOW
    out[2 * n - 1, 2 * n - 1] = Color.YELLOW
    return out


def _solve_with_existing_unique(frame: np.ndarray, sprite: np.ndarray, bg: int) -> tuple[bool, tuple[int, int, int, int] | None]:
    assignments = _find_assignments(frame, sprite, bg)
    if len(assignments) != 1:
        return False, None
    return True, assignments[0]


def _place_on_canvas(
    sprite: np.ndarray,
    frame: np.ndarray,
    sprite_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frame_h = frame.shape[0]
    min_size = frame_h + sprite_size + 6
    h = int(rng.integers(min_size, 31))
    w = int(rng.integers(min_size, 31))

    canvas = np.full((h, w), Color.BLACK, dtype=int)

    placed = False
    for _ in range(500):
        tr = int(rng.integers(0, h - frame_h + 1))
        tc = int(rng.integers(0, w - frame_h + 1))
        loc_frame = (tr, tc, tr + frame_h - 1, tc + frame_h - 1)
        # Try to put the sprite outside the 2n×2n target block.
        for _ in range(200):
            sr = int(rng.integers(0, h - sprite_size + 1))
            sc = int(rng.integers(0, w - sprite_size + 1))
            loc_sprite = (sr, sc, sr + sprite_size - 1, sc + sprite_size - 1)
            if _bbox_disjoint(loc_frame, loc_sprite):
                canvas = np.full((h, w), Color.BLACK, dtype=int)
                paste(canvas, frame, tr, tc, clip=True)
                paste(canvas, sprite, sr, sc, clip=True)
                placed = True
                break
        if placed:
            break
    assert placed, "failed to place sprite and frame without overlap"
    return canvas


def _random_sprite(shape: int, rng: np.random.Generator) -> np.ndarray:
    palette = [int(c) for c in Color.NOT_BLACK if c != Color.YELLOW]
    sprite = random_sprite(
        shape,
        shape,
        connectivity=4,
        color_palette=palette,
        background=Color.BLACK,
    )
    sprite = crop_bbox(sprite, (0, 0, shape - 1, shape - 1))
    if not np.any(sprite != Color.BLACK):
        sprite[0, 0] = int(palette[0])
    return sprite.astype(int)


def _make_distinct_rotations(sprite: np.ndarray) -> list[np.ndarray]:
    rots = _rotations(sprite)
    encoded = {rot.tobytes() for rot in rots}
    if len(encoded) < 4:
        # Force a perturbation until all rotations differ; avoids degenerate all-equal orientation cases.
        return []
    return rots


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(2000):
        n = int(rng.integers(2, 8))
        sprite = _random_sprite(n, rng)
        rots = _make_distinct_rotations(sprite)
        if not rots:
            continue

        perm = tuple(map(int, rng.permutation(4)))
        frame = _assemble_full_frame(sprite, perm)

        mode = float(rng.random())
        if mode < 0.35:
            sparse_mode = "dense"
        elif mode < 0.7:
            sparse_mode = "medium"
        else:
            sparse_mode = "sparse"

        frame_with_evidence = _draw_evidence(frame, sparse_mode, rng, n)
        bg = int(Color.BLACK)

        ok, assignment = _solve_with_existing_unique(frame_with_evidence, sprite, bg)
        if not ok or assignment is None:
            continue

        out = _assemble_output(frame_with_evidence, assignment, rots)
        # frame has only markers at the four corners; keep that convention in training input.
        input_canvas = _place_on_canvas(sprite, out, n, rng)
        # validate that main() can parse and reconstruct a deterministic output.
        try:
            _ = main(input_canvas)
        except Exception:
            continue
        return input_canvas

    raise RuntimeError("failed to generate valid input for problem 2")
