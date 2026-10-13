from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.asarray(mask, dtype=bool)
    return arr.shape, tuple(int(v) for v in arr.reshape(-1))


def _unique_d4(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    base = np.asarray(mask, dtype=bool)
    for k in range(4):
        rot = np.rot90(base, k)
        for cand in (rot, np.fliplr(rot)):
            key = _shape_key(cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand.copy())
    return out


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_shape_key(t) for t in _unique_d4(mask))


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects: list[dict] = []
    for color in sorted(set(int(v) for v in g.flat) - {bg}):
        for coords in _extract_components(g == color):
            bbox = _bbox_from_coords(coords)
            mask = _tight_mask_from_coords(coords)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "coord_set": {tuple(map(int, p)) for p in coords},
                    "bbox": bbox,
                    "mask": mask,
                    "size": int(coords.shape[0]),
                    "sig": _canonical_signature(mask),
                    "transforms": _unique_d4(mask),
                }
            )
    return bg, objects


def _dedup_placements(placements: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen = set()
    for item in placements:
        key = (item["anchor"], _shape_key(item["transform"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _completion_placements(grid: np.ndarray, bg: int, full_obj: dict, fragment_obj: dict) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frag_mask = fragment_obj["mask"]
    fh, fw = frag_mask.shape
    frag_box = fragment_obj["bbox"]
    frag_cells = fragment_obj["coord_set"]
    placements: list[dict] = []

    for transform in full_obj["transforms"]:
        th, tw = transform.shape
        if fh > th or fw > tw:
            continue
        full_cells = np.argwhere(transform)
        for off_r in range(th - fh + 1):
            for off_c in range(tw - fw + 1):
                window = transform[off_r : off_r + fh, off_c : off_c + fw]
                if not np.all(window[frag_mask]):
                    continue
                anchor_r = frag_box[0] - off_r
                anchor_c = frag_box[1] - off_c
                if anchor_r < 0 or anchor_c < 0 or anchor_r + th > h or anchor_c + tw > w:
                    continue

                ok = True
                for rr, cc in full_cells:
                    gr = anchor_r + int(rr)
                    gc = anchor_c + int(cc)
                    if (gr, gc) in frag_cells:
                        if int(g[gr, gc]) != int(fragment_obj["color"]):
                            ok = False
                            break
                    elif int(g[gr, gc]) != bg:
                        ok = False
                        break
                if ok:
                    placements.append(
                        {
                            "anchor": (int(anchor_r), int(anchor_c)),
                            "transform": transform.copy(),
                        }
                    )

    return _dedup_placements(placements)


def _candidate_solutions(grid: np.ndarray) -> list[dict]:
    bg, objects = _extract_objects(grid)
    candidates: list[dict] = []

    for frag_idx, fragment in enumerate(objects):
        remaining = [obj for i, obj in enumerate(objects) if i != frag_idx]
        counts = Counter(obj["sig"] for obj in remaining)
        if not counts:
            continue
        odd = [sig for sig, count in counts.items() if count == 1]
        if len(odd) != 1:
            continue
        if any(count not in (1, 2) for count in counts.values()):
            continue

        singleton_sig = odd[0]
        singleton_candidates = [obj for obj in remaining if obj["sig"] == singleton_sig]
        if len(singleton_candidates) != 1:
            continue
        singleton = singleton_candidates[0]
        if int(fragment["size"]) >= int(singleton["size"]):
            continue

        placements = _completion_placements(grid, bg, singleton, fragment)
        if len(placements) != 1:
            continue

        candidates.append(
            {
                "bg": int(bg),
                "singleton": singleton,
                "fragment": fragment,
                "placement": placements[0],
            }
        )

    deduped: list[dict] = []
    seen = set()
    for item in candidates:
        key = (
            item["singleton"]["color"],
            item["fragment"]["color"],
            item["placement"]["anchor"],
            _shape_key(item["placement"]["transform"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the broken mate of the unique unpaired shape using the fragment's own color.

    Input structure:
    - A dominant background surrounds several monochrome 4-connected foreground objects, each in a distinct color.
    - Almost every full object has exactly one partner of the same shape up to rotation, reflection, and translation.
    - One full object is left without a full mate, while one other object is only a connected fragment of that missing mate.

    Rule:
    - Group full objects by D4 shape equivalence and identify the unique full object whose pair is missing.
    - Find the unique placement and D4 orientation of that shape whose cells contain the fragment exactly and whose remaining cells fall on background.
    - Repaint the entire completed shape at that placement using the fragment's color, leaving everything else unchanged.

    Key insight:
    - The fragment alone is insufficient; only global shape pairing reveals which complete shape is missing and therefore how the fragment must be completed.
    """
    g = np.asarray(grid, dtype=int)
    solutions = _candidate_solutions(g)
    assert len(solutions) == 1, "expected a unique singleton/fragment completion"
    solution = solutions[0]

    out = g.copy()
    anchor_r, anchor_c = solution["placement"]["anchor"]
    transform = solution["placement"]["transform"]
    color = int(solution["fragment"]["color"])
    for rr, cc in np.argwhere(transform):
        out[anchor_r + int(rr), anchor_c + int(cc)] = color
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    comp = _extract_components(mask)
    return len(comp) == 1 and int(comp[0].shape[0]) == int(coords.shape[0])


def _random_connected_mask(
    min_area: int = 5,
    max_area: int = 9,
    min_dim: int = 3,
    max_dim: int = 6,
) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(min_dim, max_dim)
        w = random.randint(min_dim, max_dim)
        area_hi = min(max_area, h * w - 1)
        if area_hi < min_area:
            continue
        area = random.randint(min_area, area_hi)
        mask = np.zeros((h, w), dtype=bool)
        r0 = random.randrange(h)
        c0 = random.randrange(w)
        mask[r0, c0] = True
        while int(mask.sum()) < area:
            frontier = set()
            for rr, cc in np.argwhere(mask):
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True
        tight = _tight_mask_from_coords(np.argwhere(mask))
        if int(tight.sum()) != area:
            continue
        if tight.shape[0] < 2 or tight.shape[1] < 2:
            continue
        if tight.sum() == tight.size:
            continue
        return tight
    return None


def _coords_connected(coords: list[Coord]) -> bool:
    if not coords:
        return False
    pts = set(coords)
    q = deque([coords[0]])
    seen = {coords[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _subset_mask(coords: list[Coord]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    return _tight_mask_from_coords(arr)


def _embedding_records(full_transforms: list[np.ndarray], fragment_mask: np.ndarray) -> list[tuple[int, int, int]]:
    fh, fw = fragment_mask.shape
    out: list[tuple[int, int, int]] = []
    for t_idx, transform in enumerate(full_transforms):
        th, tw = transform.shape
        if fh > th or fw > tw:
            continue
        for off_r in range(th - fh + 1):
            for off_c in range(tw - fw + 1):
                window = transform[off_r : off_r + fh, off_c : off_c + fw]
                if np.all(window[fragment_mask]):
                    out.append((int(t_idx), int(off_r), int(off_c)))
    return out


def _fragment_options(full_mask: np.ndarray) -> list[dict]:
    transforms = _unique_d4(full_mask)
    options: list[dict] = []
    seen = set()
    for transform in transforms:
        coords = [tuple(map(int, p)) for p in np.argwhere(transform)]
        n = len(coords)
        for bits in range(1, 1 << n):
            size = bits.bit_count()
            if size < 3 or size >= n:
                continue
            subset = [coords[i] for i in range(n) if (bits >> i) & 1]
            if not _coords_connected(subset):
                continue
            frag = _subset_mask(subset)
            if frag.shape[0] == 1 and frag.shape[1] == 1:
                continue
            embeds = _embedding_records(transforms, frag)
            if len(embeds) != 1:
                continue
            t_idx, off_r, off_c = embeds[0]
            key = (t_idx, off_r, off_c, _shape_key(frag))
            if key in seen:
                continue
            seen.add(key)
            frag_coords = [
                (off_r + int(rr), off_c + int(cc))
                for rr, cc in np.argwhere(frag)
            ]
            options.append(
                {
                    "fragment_mask": frag,
                    "transform": transforms[t_idx].copy(),
                    "offset": (int(off_r), int(off_c)),
                    "full_coords": frag_coords,
                }
            )
    return options


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(box: BBox, h: int, w: int, gap: int) -> BBox:
    return (
        max(0, box[0] - gap),
        max(0, box[1] - gap),
        min(h - 1, box[2] + gap),
        min(w - 1, box[3] + gap),
    )


def _place_reserved_box(
    h: int,
    w: int,
    box_h: int,
    box_w: int,
    reserved: list[BBox],
    gap: int,
    tries: int = 300,
) -> Coord | None:
    for _ in range(tries):
        r0 = random.randint(0, h - box_h)
        c0 = random.randint(0, w - box_w)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        grown = _expand_bbox(box, h, w, gap)
        if any(_bbox_intersects(grown, other) for other in reserved):
            continue
        reserved.append(grown)
        return int(r0), int(c0)
    return None


def _draw_mask(grid: np.ndarray, mask: np.ndarray, anchor: Coord, color: int) -> None:
    r0, c0 = anchor
    for rr, cc in np.argwhere(mask):
        grid[r0 + int(rr), c0 + int(cc)] = int(color)


def _family_counts_without_fragment(grid: np.ndarray, fragment_color: int) -> Counter:
    _bg, objects = _extract_objects(grid)
    return Counter(obj["sig"] for obj in objects if int(obj["color"]) != int(fragment_color))


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.randint(2, 3)
    total_objects = 2 * pair_count + 2
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    if len(palette) < total_objects:
        return None
    colors = random.sample(palette, total_objects)

    singleton = _random_connected_mask(min_area=6, max_area=9)
    if singleton is None:
        return None
    fragment_candidates = _fragment_options(singleton)
    if not fragment_candidates:
        return None

    same_area_family = random.random() < 0.7
    pair_shapes: list[np.ndarray] = []
    sigs = { _canonical_signature(singleton) }
    for family_idx in range(pair_count):
        for _ in range(200):
            min_area = 5
            max_area = 9
            shape = _random_connected_mask(min_area=min_area, max_area=max_area)
            if shape is None:
                continue
            sig = _canonical_signature(shape)
            if sig in sigs:
                continue
            if same_area_family and family_idx == 0 and int(shape.sum()) != int(singleton.sum()):
                continue
            pair_shapes.append(shape)
            sigs.add(sig)
            break
        else:
            return None

    fragment_options = [
        opt
        for opt in fragment_candidates
        if _canonical_signature(opt["fragment_mask"]) not in sigs
    ]
    if not fragment_options:
        fragment_options = fragment_candidates
    fragment_pick = random.choice(fragment_options)

    h = random.randint(20, 30)
    w = random.randint(20, 30)
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    reserved: list[BBox] = []
    color_iter = iter(colors)

    # Paired full objects.
    for shape in pair_shapes:
        for _ in range(2):
            color = next(color_iter)
            transform = random.choice(_unique_d4(shape))
            th, tw = transform.shape
            anchor = _place_reserved_box(h, w, th, tw, reserved, gap=1)
            if anchor is None:
                return None
            _draw_mask(inp, transform, anchor, color)
            _draw_mask(out, transform, anchor, color)

    # The unique complete singleton.
    singleton_color = next(color_iter)
    singleton_transform = random.choice(_unique_d4(singleton))
    sh, sw = singleton_transform.shape
    singleton_anchor = _place_reserved_box(h, w, sh, sw, reserved, gap=1)
    if singleton_anchor is None:
        return None
    _draw_mask(inp, singleton_transform, singleton_anchor, singleton_color)
    _draw_mask(out, singleton_transform, singleton_anchor, singleton_color)

    # The fragment reserves the whole missing mate footprint but only shows a subset.
    fragment_color = next(color_iter)
    full_transform = fragment_pick["transform"]
    fh, fw = full_transform.shape
    fragment_anchor = _place_reserved_box(h, w, fh, fw, reserved, gap=1)
    if fragment_anchor is None:
        return None

    for rr, cc in fragment_pick["full_coords"]:
        inp[fragment_anchor[0] + rr, fragment_anchor[1] + cc] = fragment_color
    _draw_mask(out, full_transform, fragment_anchor, fragment_color)

    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(inp == bg) <= np.count_nonzero(inp != bg):
        return None

    counts = _family_counts_without_fragment(inp, fragment_color)
    if sorted(counts.values()) != [1] + [2] * pair_count:
        return None

    solutions = _candidate_solutions(inp)
    if len(solutions) != 1:
        return None
    sol = solutions[0]
    if int(sol["singleton"]["color"]) != int(singleton_color):
        return None
    if int(sol["fragment"]["color"]) != int(fragment_color):
        return None
    if tuple(sol["placement"]["anchor"]) != tuple(fragment_anchor):
        return None
    if _shape_key(sol["placement"]["transform"]) != _shape_key(full_transform):
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
