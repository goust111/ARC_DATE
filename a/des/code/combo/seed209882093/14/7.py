from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _complete_symmetric(known: np.ndarray, axis: str) -> np.ndarray | None:
    out = known.copy()
    h, w = out.shape

    for r in range(h):
        for c in range(w):
            rr, cc = (r, w - 1 - c) if axis == "v" else (h - 1 - r, c)
            a = int(out[r, c])
            b = int(out[rr, cc])
            if a != -1 and b != -1 and a != b:
                return None
            if a != -1 and b == -1:
                out[rr, cc] = a
            elif a == -1 and b != -1:
                out[r, c] = b

    if np.any(out == -1):
        return None
    if axis == "v" and not np.array_equal(out, np.fliplr(out)):
        return None
    if axis == "h" and not np.array_equal(out, np.flipud(out)):
        return None
    return out.astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a partially masked rectangular pattern by reading an identical mask copy
    and completing the hidden texture with mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 2

    mask_color = None
    pattern_color = None
    mask_infos: list[tuple[tuple[int, int, int, int], np.ndarray]] | None = None
    for color in colors:
        comps = _find_components(g == color)
        if len(comps) != 2:
            continue
        infos = []
        for comp in comps:
            bbox = _bbox_from_coords(comp)
            r0, c0, r1, c1 = bbox
            infos.append((bbox, g[r0 : r1 + 1, c0 : c1 + 1] == color))
        if infos[0][1].shape != infos[1][1].shape:
            continue
        if not np.array_equal(infos[0][1], infos[1][1]):
            continue
        mask_color = int(color)
        pattern_color = int(colors[0] if colors[1] == color else colors[1])
        mask_infos = infos
        break

    assert mask_color is not None and pattern_color is not None and mask_infos is not None

    overlap_scores = []
    for bbox, _opaque in mask_infos:
        r0, c0, r1, c1 = bbox
        overlap_scores.append(int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] == pattern_color)))
    overlay_idx = int(np.argmax(overlap_scores))
    ref_idx = 1 - overlay_idx
    assert overlap_scores[overlay_idx] > 0

    overlay_bbox = mask_infos[overlay_idx][0]
    mask_opaque = mask_infos[ref_idx][1]

    pattern_coords = np.argwhere(g == pattern_color)
    assert pattern_coords.size > 0
    pr0, pc0, pr1, pc1 = _bbox_from_coords(pattern_coords)
    patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
    known = np.where(patch == pattern_color, 1, 0)

    or0, oc0, or1, oc1 = overlay_bbox
    view = known[or0 - pr0 : or1 - pr0 + 1, oc0 - pc0 : oc1 - pc0 + 1]
    view[mask_opaque] = -1

    candidates: list[np.ndarray] = []
    for axis in ("v", "h"):
        candidate = _complete_symmetric(known, axis)
        if candidate is None:
            continue
        if not (
            candidate[0, :].all()
            and candidate[-1, :].all()
            and candidate[:, 0].all()
            and candidate[:, -1].all()
        ):
            continue
        candidates.append(candidate)

    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(np.array_equal(candidate, prev) for prev in unique):
            unique.append(candidate)
    assert len(unique) == 1

    return np.where(unique[0], pattern_color, bg).astype(int)


def _is_symmetric(mask: np.ndarray, axis: str) -> bool:
    return np.array_equal(mask, np.fliplr(mask) if axis == "v" else np.flipud(mask))


def _make_pattern_fill(ph: int, pw: int, axis: str, rng: random.Random) -> np.ndarray | None:
    for _ in range(300):
        fill = np.ones((ph, pw), dtype=bool)
        p = rng.uniform(0.18, 0.42)
        if axis == "v":
            reps = [(r, c) for r in range(1, ph - 1) for c in range(1, (pw + 1) // 2)]
        else:
            reps = [(r, c) for r in range(1, (ph + 1) // 2) for c in range(1, pw - 1)]
        rng.shuffle(reps)

        for r, c in reps:
            if rng.random() >= p:
                continue
            if axis == "v":
                fill[r, c] = False
                fill[r, pw - 1 - c] = False
            else:
                fill[r, c] = False
                fill[ph - 1 - r, c] = False

        holes = ~fill
        hole_count = int(holes[1:-1, 1:-1].sum())
        if hole_count < max(2, (ph * pw) // 12):
            continue
        if hole_count > (ph - 2) * (pw - 2) // 2:
            continue
        if axis == "v" and _is_symmetric(fill, "h"):
            continue
        if axis == "h" and _is_symmetric(fill, "v"):
            continue
        if np.any(np.all(~fill[:, 1:-1], axis=1)):
            continue
        if np.any(np.all(~fill[1:-1, :], axis=0)):
            continue
        return fill
    return None


def _build_mask(ph: int, pw: int, axis: str, rng: random.Random) -> tuple[tuple[int, int], np.ndarray] | None:
    for _ in range(300):
        mh = rng.randint(3, min(6, ph - 2))
        mw = rng.randint(3, min(6, pw - 2))

        if axis == "v":
            side = rng.choice(["left", "right"])
            r0 = rng.randint(1, ph - mh - 1)
            if side == "left":
                max_c1 = (pw - 1) // 2 if pw % 2 == 1 else pw // 2 - 1
                if max_c1 < 3:
                    continue
                c1 = rng.randint(3, max_c1)
                c0 = rng.randint(1, c1 - 2)
            else:
                min_c0 = pw // 2 + (1 if pw % 2 == 1 else 0)
                max_c0 = pw - 4
                if min_c0 > max_c0:
                    continue
                c0 = rng.randint(min_c0, max_c0)
                c1 = rng.randint(c0 + 2, pw - 2)
            mw = c1 - c0 + 1
        else:
            side = rng.choice(["top", "bottom"])
            c0 = rng.randint(1, pw - mw - 1)
            if side == "top":
                max_r1 = (ph - 1) // 2 if ph % 2 == 1 else ph // 2 - 1
                if max_r1 < 3:
                    continue
                r1 = rng.randint(3, max_r1)
                r0 = rng.randint(1, r1 - 2)
            else:
                min_r0 = ph // 2 + (1 if ph % 2 == 1 else 0)
                max_r0 = ph - 4
                if min_r0 > max_r0:
                    continue
                r0 = rng.randint(min_r0, max_r0)
                r1 = rng.randint(r0 + 2, ph - 2)
            mh = r1 - r0 + 1

        opaque = np.ones((mh, mw), dtype=bool)
        interior = [(rr, cc) for rr in range(1, mh - 1) for cc in range(1, mw - 1)]
        rng.shuffle(interior)
        hole_count = rng.randint(1, min(len(interior), max(1, (mh * mw) // 6)))
        for rr, cc in interior[:hole_count]:
            opaque[rr, cc] = False

        if len(_find_components(opaque)) != 1:
            continue
        return (int(r0), int(c0)), opaque
    return None


def _boxes_far(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    axis = rng.choice(["v", "h"])
    ph = rng.randint(6, 10)
    pw = rng.randint(6, 10)

    fill = _make_pattern_fill(ph, pw, axis, rng)
    if fill is None:
        return None

    mask_info = _build_mask(ph, pw, axis, rng)
    if mask_info is None:
        return None
    (mr0, mc0), mask_opaque = mask_info
    mh, mw = mask_opaque.shape

    colors = list(range(1, 10))
    rng.shuffle(colors)
    pattern_color, mask_color = colors[:2]
    target = np.where(fill, pattern_color, 0)

    H = min(rng.randint(max(ph, mh) + 7, max(ph, mh) + 14), 26)
    W = min(rng.randint(max(pw, mw) + 7, max(pw, mw) + 14), 26)
    if H < ph + 3 or W < pw + 3:
        return None

    for _ in range(200):
        pr0 = rng.randint(1, H - ph - 1)
        pc0 = rng.randint(1, W - pw - 1)
        pattern_box = (pr0, pc0, pr0 + ph - 1, pc0 + pw - 1)

        ref_spots = []
        for rr in range(1, H - mh):
            for cc in range(1, W - mw):
                ref_box = (rr, cc, rr + mh - 1, cc + mw - 1)
                if _boxes_far(pattern_box, ref_box, gap=1):
                    ref_spots.append((rr, cc))
        if not ref_spots:
            continue

        rr, cc = rng.choice(ref_spots)
        grid = np.zeros((H, W), dtype=int)
        grid[pr0 : pr0 + ph, pc0 : pc0 + pw] = target
        grid[pr0 + mr0 : pr0 + mr0 + mh, pc0 + mc0 : pc0 + mc0 + mw][mask_opaque] = mask_color
        grid[rr : rr + mh, cc : cc + mw][mask_opaque] = mask_color

        under = target[mr0 : mr0 + mh, mc0 : mc0 + mw]
        transparent = ~mask_opaque
        if not np.any(transparent & (under == pattern_color)):
            continue

        visible_pattern = np.argwhere(grid == pattern_color)
        if visible_pattern.size == 0 or _bbox_from_coords(visible_pattern) != pattern_box:
            continue

        try:
            solved = main(grid)
        except Exception:
            continue
        if np.array_equal(solved, target):
            return grid, target

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid masked-symmetry puzzle")
