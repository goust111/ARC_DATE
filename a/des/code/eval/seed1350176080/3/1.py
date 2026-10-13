from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _mask_connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    return len(_components(mask)) == 1


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[dict]:
    variants: list[dict] = []
    seen = set()
    for k in range(4):
        rp = np.rot90(patch, k)
        rm = np.rot90(mask, k)
        for flip in (False, True):
            vp = np.fliplr(rp) if flip else rp
            vm = np.fliplr(rm) if flip else rm
            key = (int(vm.shape[0]), int(vm.shape[1]), vm.tobytes(), vp[vm].tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "patch": np.array(vp, dtype=int, copy=True),
                    "mask": np.array(vm, dtype=bool, copy=True),
                    "rot": int(k),
                    "flip": bool(flip),
                }
            )
    return variants


def _orientation_score(
    patch: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    frame_shape: tuple[int, int],
) -> tuple[float, float]:
    vals = patch[mask]
    colors, counts = np.unique(vals, return_counts=True)
    assert len(colors) == 2
    major_color = int(colors[int(np.argmax(counts))])
    major_mask = vals == int(major_color)
    minor_mask = ~major_mask

    coords = np.argwhere(mask).astype(float)
    global_r = coords[:, 0] + float(top)
    global_c = coords[:, 1] + float(left)

    center_r = (int(frame_shape[0]) - 1) / 2.0
    center_c = (int(frame_shape[1]) - 1) / 2.0
    dist2 = (global_r - center_r) ** 2 + (global_c - center_c) ** 2

    hole_r = float(global_r.mean())
    hole_c = float(global_c.mean())
    vec_r = center_r - hole_r
    vec_c = center_c - hole_c
    proj = (global_r - hole_r) * vec_r + (global_c - hole_c) * vec_c

    dist_gap = float(dist2[minor_mask].mean() - dist2[major_mask].mean())
    proj_gap = float(proj[major_mask].mean() - proj[minor_mask].mean())
    return dist_gap, proj_gap


def _score_better(a: tuple[float, float], b: tuple[float, float], eps: float = 1e-9) -> bool:
    return a[0] > b[0] + eps or (abs(a[0] - b[0]) <= eps and a[1] > b[1] + eps)


def _score_equal(a: tuple[float, float], b: tuple[float, float], eps: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _paste_mask(out: np.ndarray, patch: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    h, w = mask.shape
    view = out[int(top) : int(top) + int(h), int(left) : int(left) + int(w)]
    view[mask] = patch[mask]


def _canonical_mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    keys = []
    for k in range(4):
        m = np.rot90(mask, k)
        keys.append((int(m.shape[0]), int(m.shape[1]), m.tobytes()))
        f = np.fliplr(m)
        keys.append((int(f.shape[0]), int(f.shape[1]), f.tobytes()))
    return min(keys)


def _random_polyomino(area: int, max_span: int = 4, tries: int = 80) -> np.ndarray | None:
    for _ in range(int(tries)):
        cells = {(0, 0)}
        while len(cells) < int(area):
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    frontier.add((int(r + dr), int(c + dc)))
            frontier -= cells
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            cells.add((int(nr), int(nc)))
        if len(cells) != int(area):
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        norm = sorted((int(r - r0), int(c - c0)) for r, c in cells)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h > int(max_span) or w > int(max_span):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[int(r), int(c)] = True
        return mask
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = mask.copy()
    for dr, dc in DIR8:
        src_r0 = max(0, -dr)
        src_r1 = min(h, h - dr)
        src_c0 = max(0, -dc)
        src_c1 = min(w, w - dc)
        dst_r0 = max(0, dr)
        dst_r1 = min(h, h + dr)
        dst_c0 = max(0, dc)
        dst_c1 = min(w, w + dc)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                out.append((rr, cc))
                break
    return out


def _subset_candidates(mask: np.ndarray, size: int) -> list[tuple[tuple[int, int], ...]]:
    area = int(mask.sum())
    if area <= int(size) * 2:
        return []
    boundary = _boundary_cells(mask)
    boundary_set = set(boundary)
    out: list[tuple[tuple[int, int], ...]] = []
    if int(size) == 1:
        for cell in boundary:
            rem = mask.copy()
            rem[cell] = False
            if _mask_connected(rem):
                out.append((cell,))
        return out

    seen = set()
    for r, c in boundary:
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if (nr, nc) not in boundary_set:
                continue
            subset = tuple(sorted(((int(r), int(c)), (int(nr), int(nc)))))
            if subset in seen:
                continue
            seen.add(subset)
            rem = mask.copy()
            for cell in subset:
                rem[cell] = False
            if _mask_connected(rem):
                out.append(subset)
    return out


def _build_colored_patch(
    mask: np.ndarray,
    subset: tuple[tuple[int, int], ...],
    major_color: int,
    minor_color: int,
) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(major_color)
    for r, c in subset:
        patch[int(r), int(c)] = int(minor_color)
    return patch


def _identity_is_unique_best(
    patch: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    frame_shape: tuple[int, int],
) -> tuple[float, float] | None:
    best_score: tuple[float, float] | None = None
    best_is_identity = False
    best_count = 0

    for variant in _dihedral_variants(patch, mask):
        vm = variant["mask"]
        if vm.shape != mask.shape or not np.array_equal(vm, mask):
            continue
        score = _orientation_score(variant["patch"], vm, int(top), int(left), frame_shape)
        is_identity = np.array_equal(variant["patch"], patch)
        if best_score is None or _score_better(score, best_score):
            best_score = score
            best_is_identity = bool(is_identity)
            best_count = 1
        elif _score_equal(score, best_score):
            best_count += 1
            best_is_identity = best_is_identity or bool(is_identity)

    if best_score is None:
        return None
    if not best_is_identity or best_count != 1:
        return None
    if not _score_better(best_score, (0.0, 0.0)):
        return None
    return best_score


def _choose_coloring(
    mask: np.ndarray,
    top: int,
    left: int,
    frame_shape: tuple[int, int],
    major_color: int,
    minor_color: int,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]] | None:
    preferred = random.choice([1, 2])
    sizes = [preferred, 1 if preferred == 2 else 2]
    for size in sizes:
        candidates = []
        for subset in _subset_candidates(mask, int(size)):
            patch = _build_colored_patch(mask, subset, int(major_color), int(minor_color))
            score = _identity_is_unique_best(patch, mask, int(top), int(left), frame_shape)
            if score is None:
                continue
            candidates.append((score, patch, subset))
        if candidates:
            candidates.sort(key=lambda item: (item[0][0], item[0][1]), reverse=True)
            pick_pool = candidates[: min(3, len(candidates))]
            _, patch, subset = random.choice(pick_pool)
            return patch, subset
    return None


def _rects_overlap_with_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    return not (
        a[2] + int(gap) <= b[0]
        or b[2] + int(gap) <= a[0]
        or a[3] + int(gap) <= b[1]
        or b[3] + int(gap) <= a[1]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame hole to an outside two-color piece and orient it so the
    majority color points toward the frame center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    comps = _components(g != int(bg))
    assert comps
    comps.sort(key=lambda coords: int(coords.shape[0]), reverse=True)

    frame_coords = comps[0]
    fr0, fc0, fr1, fc1 = _bbox_coords(frame_coords)
    frame_colors = {int(x) for x in g[frame_coords[:, 0], frame_coords[:, 1]].tolist()}
    assert len(frame_colors) == 1
    frame_color = int(next(iter(frame_colors)))

    frame_crop = g[int(fr0) : int(fr1) + 1, int(fc0) : int(fc1) + 1]
    fh, fw = map(int, frame_crop.shape)

    holes: list[dict] = []
    for coords in _components(frame_crop == int(bg)):
        if any(
            int(r) == 0 or int(c) == 0 or int(r) == int(fh - 1) or int(c) == int(fw - 1)
            for r, c in coords
        ):
            continue
        mask, (r0, c0, _r1, _c1) = _tight_mask(coords)
        holes.append(
            {
                "top": int(r0),
                "left": int(c0),
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    assert holes

    pieces: list[dict] = []
    for coords in comps[1:]:
        pr0, pc0, pr1, pc1 = _bbox_coords(coords)
        mask = np.zeros((int(pr1 - pr0 + 1), int(pc1 - pc0 + 1)), dtype=bool)
        rr = coords[:, 0] - pr0
        cc = coords[:, 1] - pc0
        mask[rr, cc] = True
        patch = np.full(mask.shape, int(bg), dtype=int)
        patch[rr, cc] = g[coords[:, 0], coords[:, 1]]
        vals = patch[mask]
        colors, counts = np.unique(vals, return_counts=True)
        if len(colors) != 2 or int(counts[0]) == int(counts[1]):
            continue
        pieces.append({"patch": patch, "mask": mask, "variants": _dihedral_variants(patch, mask)})
    assert len(pieces) >= len(holes)

    for hole in holes:
        candidates = []
        for piece_index, piece in enumerate(pieces):
            best_variant = None
            best_score = None
            for variant in piece["variants"]:
                vm = variant["mask"]
                if vm.shape != hole["mask"].shape or not np.array_equal(vm, hole["mask"]):
                    continue
                score = _orientation_score(
                    variant["patch"],
                    vm,
                    int(hole["top"]),
                    int(hole["left"]),
                    (int(fh), int(fw)),
                )
                if best_score is None or _score_better(score, best_score):
                    best_score = score
                    best_variant = variant
            if best_variant is not None:
                candidates.append(
                    {
                        "piece": int(piece_index),
                        "score": best_score,
                        "patch": best_variant["patch"],
                    }
                )
        assert candidates
        hole["candidates"] = candidates

    order = sorted(
        range(len(holes)),
        key=lambda idx: (
            len(holes[idx]["candidates"]),
            -int(holes[idx]["area"]),
            int(holes[idx]["top"]),
            int(holes[idx]["left"]),
        ),
    )
    used: set[int] = set()
    assignment: dict[int, dict] = {}

    def search(k: int) -> bool:
        if int(k) == len(order):
            return True
        hole_index = int(order[k])
        candidates = sorted(
            holes[hole_index]["candidates"],
            key=lambda cand: (
                -cand["score"][0],
                -cand["score"][1],
                int(cand["piece"]),
            ),
        )
        for cand in candidates:
            piece_index = int(cand["piece"])
            if piece_index in used:
                continue
            used.add(piece_index)
            assignment[hole_index] = cand
            if search(int(k + 1)):
                return True
            used.remove(piece_index)
            del assignment[hole_index]
        return False

    assert search(0)

    out = np.full((int(fh), int(fw)), int(frame_color), dtype=int)
    for hole_index, hole in enumerate(holes):
        chosen = assignment[hole_index]
        _paste_mask(out, chosen["patch"], int(hole["top"]), int(hole["left"]), hole["mask"])
    return out


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    n_holes = random.randint(2, 4)
    frame_h = random.randint(10, 16)
    frame_w = random.randint(10, 16)
    if int(frame_h * frame_w) < 90:
        frame_h += 1
        frame_w += 1

    color_pool = list(range(10))
    random.shuffle(color_pool)
    needed = 2 + 2 * int(n_holes)
    if len(color_pool) < needed:
        return None
    bg = int(color_pool[0])
    frame_color = int(color_pool[1])
    piece_colors = color_pool[2 : 2 + 2 * int(n_holes)]

    output = np.full((int(frame_h), int(frame_w)), int(frame_color), dtype=int)
    frame_shape = (int(frame_h), int(frame_w))

    forbidden = np.zeros((int(frame_h), int(frame_w)), dtype=bool)
    forbidden[0, :] = True
    forbidden[-1, :] = True
    forbidden[:, 0] = True
    forbidden[:, -1] = True

    used_shape_keys = set()
    holes: list[dict] = []

    for hole_index in range(int(n_holes)):
        major_color = int(piece_colors[2 * hole_index])
        minor_color = int(piece_colors[2 * hole_index + 1])
        placed = False
        for _ in range(120):
            area = random.randint(3, 7)
            mask = _random_polyomino(area, max_span=4)
            if mask is None:
                continue
            key = _canonical_mask_key(mask)
            if key in used_shape_keys:
                continue
            mh, mw = map(int, mask.shape)
            positions = []
            for top in range(1, int(frame_h - mh)):
                for left in range(1, int(frame_w - mw)):
                    if np.any(forbidden[top : top + mh, left : left + mw] & mask):
                        continue
                    positions.append((int(top), int(left)))
            random.shuffle(positions)
            for top, left in positions[:50]:
                chosen = _choose_coloring(
                    mask,
                    int(top),
                    int(left),
                    frame_shape,
                    int(major_color),
                    int(minor_color),
                )
                if chosen is None:
                    continue
                patch, subset = chosen
                _paste_mask(output, patch, int(top), int(left), mask)
                placed_mask = np.zeros_like(forbidden)
                placed_mask[int(top) : int(top) + int(mh), int(left) : int(left) + int(mw)] = mask
                forbidden |= _dilate8(placed_mask)
                holes.append(
                    {
                        "top": int(top),
                        "left": int(left),
                        "mask": np.array(mask, dtype=bool, copy=True),
                        "patch": np.array(patch, dtype=int, copy=True),
                        "subset": tuple(subset),
                    }
                )
                used_shape_keys.add(key)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    frame_input = np.full((int(frame_h), int(frame_w)), int(frame_color), dtype=int)
    for hole in holes:
        mh, mw = hole["mask"].shape
        frame_view = frame_input[
            int(hole["top"]) : int(hole["top"]) + int(mh),
            int(hole["left"]) : int(hole["left"]) + int(mw),
        ]
        frame_view[hole["mask"]] = int(bg)

    transformed_pieces = []
    for hole in holes:
        variants = _dihedral_variants(hole["patch"], hole["mask"])
        choice = random.choice(variants)
        transformed_pieces.append(
            {
                "patch": choice["patch"],
                "mask": choice["mask"],
            }
        )

    nonbg_area = int(np.sum(frame_input != int(bg))) + sum(
        int(piece["mask"].sum()) for piece in transformed_pieces
    )
    min_h = max(20, int(frame_h + 6))
    min_w = max(20, int(frame_w + 6))
    canvas_sizes = [
        (h, w)
        for h in range(int(min_h), 31)
        for w in range(int(min_w), 31)
        if h * w > 2 * int(nonbg_area)
    ]
    if not canvas_sizes:
        return None

    for _ in range(120):
        canvas_h, canvas_w = random.choice(canvas_sizes)
        if canvas_h - int(frame_h) <= 2 or canvas_w - int(frame_w) <= 2:
            continue

        frame_top = random.randint(1, int(canvas_h - frame_h - 1))
        frame_left = random.randint(1, int(canvas_w - frame_w - 1))
        rects = [(int(frame_top), int(frame_left), int(frame_top + frame_h), int(frame_left + frame_w))]
        placements: list[tuple[int, int]] = []

        order = sorted(
            range(len(transformed_pieces)),
            key=lambda idx: int(transformed_pieces[idx]["mask"].sum()),
            reverse=True,
        )
        ok = True
        for piece_index in order:
            piece = transformed_pieces[piece_index]
            ph, pw = piece["patch"].shape
            positions = []
            for top in range(1, int(canvas_h - ph)):
                for left in range(1, int(canvas_w - pw)):
                    rect = (int(top), int(left), int(top + ph), int(left + pw))
                    if any(_rects_overlap_with_gap(rect, other, gap=1) for other in rects):
                        continue
                    positions.append((int(top), int(left)))
            if not positions:
                ok = False
                break
            pos = random.choice(positions)
            placements.append((int(piece_index), int(pos[0]), int(pos[1])))
            rects.append((int(pos[0]), int(pos[1]), int(pos[0] + ph), int(pos[1] + pw)))
        if not ok:
            continue

        inp = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
        inp[
            int(frame_top) : int(frame_top) + int(frame_h),
            int(frame_left) : int(frame_left) + int(frame_w),
        ] = frame_input
        for piece_index, top, left in placements:
            piece = transformed_pieces[int(piece_index)]
            _paste_mask(inp, piece["patch"], int(top), int(left), piece["mask"])

        counts = Counter(map(int, inp.ravel()))
        most_common = counts.most_common()
        if len(most_common) < 2 or most_common[0][0] != int(bg) or most_common[0][1] <= most_common[1][1]:
            continue

        comps = _components(inp != int(bg))
        if len(comps) != 1 + len(holes):
            continue

        actual = main(inp)
        if np.array_equal(actual, output):
            return inp, output

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            inp, _out = scene
            return inp
    raise RuntimeError("failed to generate a valid input")
