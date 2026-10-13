from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
MAX_SIZE = 30
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SYMMETRY_KINDS = ("rot4", "mirror_h", "mirror_v", "diag")
FRAME_COUNT_OPTIONS = (1, 2, 3, 4)
FRAME_COUNT_WEIGHTS = (1, 3, 3, 1)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _neighbors8(r: int, c: int, shape: tuple[int, int]):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                yield rr, cc


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, tuple[int, int, int, int]]]:
    comps = _components8(grid == FRAME)
    frames: list[tuple[int, int, int, int, tuple[int, int, int, int]]] = []
    for comp in comps:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame bbox must leave room for four corners"

        rel = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            rel[r - r0, c - c0] = True
        expected = np.zeros((h, w), dtype=bool)
        expected[0, 1:-1] = True
        expected[-1, 1:-1] = True
        expected[1:-1, 0] = True
        expected[1:-1, -1] = True
        assert np.array_equal(rel, expected), "grey component must be a one-pixel rectangular border without corners"

        corners = (
            int(grid[r0, c0]),
            int(grid[r0, c1]),
            int(grid[r1, c0]),
            int(grid[r1, c1]),
        )
        assert all(color not in (BG, FRAME) for color in corners), "frame corners must be colored markers"
        frames.append((r0, c0, r1, c1, corners))

    assert frames, "expected at least one grey frame"
    frames.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return frames


def _decode_symmetry(corners: tuple[int, int, int, int]) -> str:
    tl, tr, bl, br = corners
    if tl == tr == bl == br:
        return "rot4"
    if tl == tr and bl == br and tl != bl:
        return "mirror_h"
    if tl == bl and tr == br and tl != tr:
        return "mirror_v"
    if tl == br and tr == bl and tl != tr:
        return "diag"
    raise AssertionError(f"unsupported corner code: {corners}")


def _orbit(kind: str, h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    if kind == "rot4":
        assert h == w, "90-degree rotation requires a square interior"
        pts = set()
        rr, cc = r, c
        for _ in range(4):
            pts.add((int(rr), int(cc)))
            rr, cc = cc, w - 1 - rr
        return tuple(sorted(pts))
    if kind == "mirror_h":
        return tuple(sorted({(int(r), int(c)), (h - 1 - int(r), int(c))}))
    if kind == "mirror_v":
        return tuple(sorted({(int(r), int(c)), (int(r), w - 1 - int(c))}))
    if kind == "diag":
        assert h == w, "main-diagonal reflection requires a square interior"
        return tuple(sorted({(int(r), int(c)), (int(c), int(r))}))
    raise AssertionError(kind)


def _all_orbits(kind: str, h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[frozenset[tuple[int, int]]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            orbit = _orbit(kind, h, w, r, c)
            key = frozenset(orbit)
            if key in seen:
                continue
            seen.add(key)
            orbits.append(orbit)
    random.shuffle(orbits)
    return orbits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the corner-marker symmetry of each grey frame and complete the partial pattern inside it.

    Input structure:
    - The background is black.
    - One or more isolated rectangular frames are drawn with a one-pixel-thick grey border, while each of the four frame corners is recolored as a non-grey marker.
    - Inside each frame, a sparse set of colored seed pixels shows an incomplete symmetric pattern.

    Rule:
    - Read the four corner-marker colors of each frame to determine its symmetry type: all equal means fourfold rotation, equal top corners versus equal bottom corners means horizontal-axis mirroring, equal left corners versus equal right corners means vertical-axis mirroring, and equal main-diagonal corners means main-diagonal reflection.
    - For every interior seed pixel, map it through the corresponding symmetry orbit and paint every missing orbit mate with the same color.
    - Keep the grey border and the colored corner markers unchanged.

    Key insight:
    - The corner colors are not part of the drawable motif; they are a code that chooses which symmetry transform must be closed over the interior seeds.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG

    out = g.copy()
    for r0, c0, r1, c1, corners in _find_frames(g):
        kind = _decode_symmetry(corners)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        h, w = map(int, inner.shape)
        if kind in {"rot4", "diag"}:
            assert h == w, "rotation and diagonal frames must be square inside"

        seeds = np.argwhere(inner != bg)
        assert seeds.size > 0, "each frame must contain at least one seed pixel"
        for rr, cc in seeds:
            color = int(inner[rr, cc])
            for orr, occ in _orbit(kind, h, w, int(rr), int(cc)):
                ar = r0 + 1 + orr
                ac = c0 + 1 + occ
                assert int(out[ar, ac]) in {bg, color}, "symmetry orbit must not introduce conflicting colors"
                out[ar, ac] = color

    return out


def _choose_frame_count() -> int:
    return random.choices(FRAME_COUNT_OPTIONS, weights=FRAME_COUNT_WEIGHTS, k=1)[0]


def _choose_dims(kind: str, frame_count: int) -> tuple[int, int]:
    high = 8 if frame_count <= 2 else 7 if frame_count == 3 else 6
    if kind in {"rot4", "diag"}:
        side = random.randint(3, high)
        return side, side

    h = random.randint(3, high)
    w = random.randint(3, high + 1)
    if random.random() < 0.65 and h == w:
        if w < high + 1:
            w += 1
        elif h > 3:
            h -= 1
    return h, w


def _corner_code(kind: str) -> tuple[int, int, int, int]:
    if kind == "rot4":
        color = random.choice(MARKER_COLORS)
        return (color, color, color, color)

    a, b = random.sample(MARKER_COLORS, 2)
    if kind == "mirror_h":
        return (a, a, b, b)
    if kind == "mirror_v":
        return (a, b, a, b)
    if kind == "diag":
        return (a, b, b, a)
    raise AssertionError(kind)


def _make_inner_pair(kind: str, h: int, w: int, forbidden: set[int]) -> tuple[np.ndarray, np.ndarray]:
    orbits = _all_orbits(kind, h, w)
    nontrivial = [orbit for orbit in orbits if len(orbit) > 1]
    assert nontrivial, "each generated frame must support a nontrivial symmetry orbit"

    first = random.choice(nontrivial)
    max_fill = max(len(first), int(round(h * w * random.uniform(0.18, 0.45))))
    target_orbits = random.randint(1, min(6, len(orbits)))

    selected = [first]
    filled = len(first)
    for orbit in orbits:
        if orbit == first or len(selected) >= target_orbits:
            continue
        if filled + len(orbit) > max_fill and filled >= max(2, int(np.ceil(h * w * 0.15))):
            continue
        take_prob = 0.65 if len(orbit) > 1 else 0.4
        if random.random() < take_prob:
            selected.append(orbit)
            filled += len(orbit)

    if len(selected) == 1:
        for orbit in orbits:
            if orbit != first:
                selected.append(orbit)
                break

    palette = [color for color in MARKER_COLORS if color not in forbidden]
    palette_size = random.randint(1, min(4, len(palette), len(selected)))
    chosen_palette = random.sample(palette, palette_size)

    full = np.full((h, w), BG, dtype=int)
    for orbit in selected:
        color = random.choice(chosen_palette)
        for r, c in orbit:
            full[r, c] = color

    inp = full.copy()
    shrinkable = [orbit for orbit in selected if len(orbit) > 1]
    partial_count = random.randint(1, len(shrinkable))
    random.shuffle(shrinkable)
    for i, orbit in enumerate(shrinkable):
        if i >= partial_count and random.random() < 0.5:
            continue
        keep_count = random.randint(1, len(orbit) - 1)
        keep = set(random.sample(list(orbit), keep_count))
        for r, c in orbit:
            if (r, c) not in keep:
                inp[r, c] = BG

    assert np.any(inp != full), "input must omit at least one symmetric mate"
    return inp, full


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (
        a[2] + 1 < b[0]
        or b[2] + 1 < a[0]
        or a[3] + 1 < b[1]
        or b[3] + 1 < a[1]
    )


def _stamp_frame(canvas: np.ndarray, top: int, left: int, corners: tuple[int, int, int, int], inner: np.ndarray) -> None:
    h, w = map(int, inner.shape)
    bottom = top + h + 1
    right = left + w + 1

    assert np.all(canvas[top : bottom + 1, left : right + 1] == BG), "frames must not overlap"
    canvas[top, left + 1 : right] = FRAME
    canvas[bottom, left + 1 : right] = FRAME
    canvas[top + 1 : bottom, left] = FRAME
    canvas[top + 1 : bottom, right] = FRAME
    canvas[top, left] = corners[0]
    canvas[top, right] = corners[1]
    canvas[bottom, left] = corners[2]
    canvas[bottom, right] = corners[3]
    canvas[top + 1 : bottom, left + 1 : right] = inner


def _sample_specs() -> list[dict]:
    frame_count = _choose_frame_count()
    kinds = random.sample(list(SYMMETRY_KINDS), frame_count)
    specs = []
    for kind in kinds:
        h, w = _choose_dims(kind, frame_count)
        corners = _corner_code(kind)
        inp_inner, out_inner = _make_inner_pair(kind, h, w, set(corners))
        specs.append(
            {
                "kind": kind,
                "corners": corners,
                "inp_inner": inp_inner,
                "out_inner": out_inner,
                "outer_h": h + 2,
                "outer_w": w + 2,
            }
        )
    random.shuffle(specs)
    return specs


def _place_specs(specs: list[dict]) -> list[tuple[dict, tuple[int, int, int, int]]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    placed: list[tuple[dict, tuple[int, int, int, int]]] = []
    for spec in sorted(specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        ok = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, MAX_SIZE - spec["outer_h"] - 1)
            left = random.randint(1, MAX_SIZE - spec["outer_w"] - 1)
            box = (top, left, top + spec["outer_h"] - 1, left + spec["outer_w"] - 1)
            if all(_boxes_far_enough(box, other) for other in boxes):
                boxes.append(box)
                placed.append((spec, box))
                ok = True
                break
        if not ok:
            return None
    return placed


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(out != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_pad = random.randint(0, int(r0))
    left_pad = random.randint(0, int(c0))
    bottom_pad = random.randint(0, int(out.shape[0] - 1 - r1))
    right_pad = random.randint(0, int(out.shape[1] - 1 - c1))
    return (
        inp[r0 - top_pad : r1 + bottom_pad + 1, c0 - left_pad : c1 + right_pad + 1],
        out[r0 - top_pad : r1 + bottom_pad + 1, c0 - left_pad : c1 + right_pad + 1],
    )


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    placed = _place_specs(specs)
    if placed is None:
        return None

    inp = np.full((MAX_SIZE, MAX_SIZE), BG, dtype=int)
    out = np.full((MAX_SIZE, MAX_SIZE), BG, dtype=int)
    for spec, (top, left, _bottom, _right) in placed:
        _stamp_frame(inp, top, left, spec["corners"], spec["inp_inner"])
        _stamp_frame(out, top, left, spec["corners"], spec["out_inner"])

    inp, out = _crop_pair(inp, out)
    if inp.shape[0] > MAX_SIZE or inp.shape[1] > MAX_SIZE:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-frame ARC input")
