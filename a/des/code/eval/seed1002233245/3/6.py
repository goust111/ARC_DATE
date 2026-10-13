from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


FRAME_COLOR = 9
FRAGMENT_COLOR = 5

GRID_H_RANGE = (24, 32)
GRID_W_RANGE = (24, 32)
MATCHED_RANGE = (1, 3)
UNMATCHED_RANGE = (0, 2)
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def component_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    t = trim_mask(mask)
    return int(t.shape[0]), int(t.shape[1]), t.tobytes()


def dedupe_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for mask in masks:
        t = trim_mask(mask)
        key = shape_key(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(trim_mask(rot))
        variants.append(trim_mask(np.fliplr(rot)))
    return dedupe_masks(variants)


def rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    return dedupe_masks([trim_mask(np.rot90(mask, k)) for k in range(4)])


def reflection_only_variants(mask: np.ndarray) -> list[np.ndarray]:
    rot_keys = {shape_key(v) for v in rotation_variants(mask)}
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        v = trim_mask(np.fliplr(np.rot90(mask, k)))
        key = shape_key(v)
        if key in rot_keys or key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def canonical_shape(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(shape_key(v) for v in dihedral_variants(mask))


def extract_holes(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    holes: list[np.ndarray] = []
    bg = ~mask
    for comp in components_mask(bg):
        if np.any(
            (comp[:, 0] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == 0)
            | (comp[:, 1] == w - 1)
        ):
            continue
        holes.append(component_to_mask(comp))
    return holes


def is_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask)
    return len(comps) == 1


def is_rect_frame_component(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    candidates: dict[int, list[dict[str, object]]] = {}
    for color in [int(c) for c in np.unique(grid) if int(c) != bg]:
        frames: list[dict[str, object]] = []
        for comp in components_mask(grid == color):
            comp_mask = component_to_mask(comp)
            if not is_rect_frame_component(comp_mask):
                continue
            frames.append({"bbox": bbox_coords(comp), "color": color})
        if frames:
            frames.sort(key=lambda d: tuple(d["bbox"]))  # type: ignore[arg-type]
            candidates[color] = frames
    if not candidates:
        raise AssertionError("no frame components found")
    frame_color, frames = max(
        candidates.items(),
        key=lambda kv: (len(kv[1]), sum((b["bbox"][2] - b["bbox"][0] + 1) * (b["bbox"][3] - b["bbox"][1] + 1) for b in kv[1])),
    )
    return int(frame_color), frames


def key_from_frame(
    grid: np.ndarray, bg: int, frame_color: int, bbox: tuple[int, int, int, int]
) -> tuple[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(c) for c in np.unique(interior) if int(c) not in (bg, frame_color)]
    assert len(colors) == 1, "each frame must contain exactly one key color"
    key_color = colors[0]
    comps = components_mask(interior == key_color)
    assert len(comps) == 1, "key must be a single connected component"
    return int(key_color), component_to_mask(comps[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match grey fragments to lock keys by internal hole shape, then move and recolor them.

    Input structure:
    - bg is the most common color.
    - Grey connected fragments lie outside maroon rectangular frames.
    - Each frame contains one colored key shape whose geometry matches one fragment hole
      up to rotation or reflection.

    Rule:
    - Extract every fragment hole.
    - Match each key to a unique fragment hole under the D4 symmetry group.
    - Clear all scattered fragments.
    - Keep the frames, erase the keys, and place the matched fragment centered in the
      corresponding frame interior using the key color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_color, frames = find_frames(g, bg)
    frame_region = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        frame_region[r0 : r1 + 1, c0 : c1 + 1] = True

    outside_colors = [
        int(c)
        for c in np.unique(g[~frame_region])
        if int(c) not in (bg, frame_color)
    ]
    assert outside_colors, "no fragment color found outside frames"
    fragment_color = max(
        set(outside_colors),
        key=lambda color: int(np.count_nonzero((g == color) & ~frame_region)),
    )

    fragments: list[dict[str, object]] = []
    fragment_mask = (g == fragment_color) & ~frame_region
    for comp in components_mask(fragment_mask):
        mask = component_to_mask(comp)
        fragments.append({"mask": mask, "holes": extract_holes(mask)})

    hole_index: dict[tuple[int, int, bytes], list[tuple[int, int]]] = defaultdict(list)
    for fi, fragment in enumerate(fragments):
        for hi, hole in enumerate(fragment["holes"]):  # type: ignore[index]
            hole_index[canonical_shape(hole)].append((fi, hi))

    out = g.copy()
    out[g == fragment_color] = bg
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    used_fragments: set[int] = set()
    for frame in frames:
        bbox = frame["bbox"]  # type: ignore[assignment]
        key_color, key_mask = key_from_frame(g, bg, frame_color, bbox)
        candidates = [cand for cand in hole_index[canonical_shape(key_mask)] if cand[0] not in used_fragments]
        assert len(candidates) == 1, "each key must match one unique fragment"
        frag_idx, _hole_idx = candidates[0]
        used_fragments.add(frag_idx)

        frag_mask = fragments[frag_idx]["mask"]  # type: ignore[index]
        fh, fw = frag_mask.shape
        r0, c0, r1, c1 = bbox
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert fh <= inner_h and fw <= inner_w, "fragment must fit inside its frame"
        top = r0 + 1 + (inner_h - fh) // 2
        left = c0 + 1 + (inner_w - fw) // 2
        out[top : top + fh, left : left + fw][frag_mask] = key_color

    return out


def random_connected_shape(
    min_area: int = 2,
    max_area: int = 5,
    max_side: int = 4,
    require_transforms: bool = False,
) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, max_side)
        w = random.randint(2, max_side)
        area_hi = min(max_area, h * w - 1)
        if area_hi < min_area:
            continue
        target = random.randint(min_area, area_hi)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier: set[tuple[int, int]] = set(neighbors4(start[0], start[1], h, w))
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            rr, cc = cell
            if mask[rr, cc]:
                continue
            mask[rr, cc] = True
            for nb in neighbors4(rr, cc, h, w):
                if not mask[nb]:
                    frontier.add(nb)
        if int(mask.sum()) != target:
            continue
        mask = trim_mask(mask)
        if require_transforms:
            rots = rotation_variants(mask)
            flips = reflection_only_variants(mask)
            if len(rots) < 2 or not flips:
                continue
        return mask
    return None


def unique_hole_shape(
    used: set[tuple[int, int, bytes]], require_transforms: bool
) -> np.ndarray | None:
    for _ in range(300):
        mask = random_connected_shape(require_transforms=require_transforms)
        if mask is None:
            continue
        key = canonical_shape(mask)
        if key in used:
            continue
        used.add(key)
        return mask
    return None


def choose_key_variant(hole_mask: np.ndarray, mode: str) -> np.ndarray | None:
    same = trim_mask(hole_mask)
    rots = [v for v in rotation_variants(hole_mask) if shape_key(v) != shape_key(same)]
    flips = reflection_only_variants(hole_mask)
    all_variants = dihedral_variants(hole_mask)
    if mode == "same" and shape_key(same):
        return same
    if mode == "rot" and rots:
        return random.choice(rots)
    if mode == "flip" and flips:
        return random.choice(flips)
    if not all_variants:
        return None
    return random.choice(all_variants)


def sample_modes(n: int) -> list[str]:
    modes = (["same", "rot", "flip"] * ((n + 2) // 3))[:n]
    random.shuffle(modes)
    return modes


def make_fragment(hole_masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    wanted = sorted(canonical_shape(h) for h in hole_masks)
    max_dim = max(max(mask.shape) for mask in hole_masks)
    for _ in range(200):
        base_h = random.randint(max_dim + 4, max_dim + 7)
        base_w = random.randint(max_dim + 4, max_dim + 7)

        fragment = np.ones((base_h, base_w), dtype=bool)
        blocked = np.zeros((base_h, base_w), dtype=bool)
        ok = True

        for hole in sorted(hole_masks, key=lambda m: (-int(m.sum()), m.shape[0], m.shape[1])):
            hh, hw = hole.shape
            candidates: list[tuple[int, int]] = []
            for top in range(1, base_h - hh):
                for left in range(1, base_w - hw):
                    r0 = max(0, top - 1)
                    c0 = max(0, left - 1)
                    r1 = min(base_h, top + hh + 1)
                    c1 = min(base_w, left + hw + 1)
                    if blocked[r0:r1, c0:c1].any():
                        continue
                    candidates.append((top, left))
            if not candidates:
                ok = False
                break
            top, left = random.choice(candidates)
            fragment[top : top + hh, left : left + hw][hole] = False
            blocked[max(0, top - 1) : min(base_h, top + hh + 1), max(0, left - 1) : min(base_w, left + hw + 1)] = True

        if not ok:
            continue

        pad = 2
        canvas = np.zeros((base_h + 2 * pad, base_w + 2 * pad), dtype=bool)
        support = np.zeros_like(canvas)
        canvas[pad : pad + base_h, pad : pad + base_w] = fragment
        support[pad : pad + base_h, pad : pad + base_w] = True

        sides = ["top", "bottom", "left", "right"]
        random.shuffle(sides)
        for side in sides[: random.randint(0, 4)]:
            if side in ("top", "bottom"):
                thickness = random.randint(1, pad)
                length = random.randint(max(2, base_w // 3), base_w)
                start = random.randint(0, base_w - length)
                cols = slice(pad + start, pad + start + length)
                if side == "top":
                    rows = slice(pad - thickness, pad)
                else:
                    rows = slice(pad + base_h, pad + base_h + thickness)
            else:
                thickness = random.randint(1, pad)
                length = random.randint(max(2, base_h // 3), base_h)
                start = random.randint(0, base_h - length)
                rows = slice(pad + start, pad + start + length)
                if side == "left":
                    cols = slice(pad - thickness, pad)
                else:
                    cols = slice(pad + base_w, pad + base_w + thickness)
            canvas[rows, cols] = True

        pts = np.argwhere(canvas)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        fragment_mask = canvas[r0 : r1 + 1, c0 : c1 + 1].copy()
        support_mask = support[r0 : r1 + 1, c0 : c1 + 1].copy()

        if not is_connected(fragment_mask):
            continue
        got = sorted(canonical_shape(h) for h in extract_holes(fragment_mask))
        if got != wanted:
            continue
        return fragment_mask, support_mask

    return None


def pick_placement(
    reserved: np.ndarray, obj_h: int, obj_w: int, pad: int = 1
) -> tuple[int, int] | None:
    h, w = reserved.shape
    candidates: list[tuple[int, int]] = []
    for top in range(1, h - obj_h):
        for left in range(1, w - obj_w):
            r0 = top - pad
            c0 = left - pad
            r1 = top + obj_h + pad
            c1 = left + obj_w + pad
            if reserved[r0:r1, c0:c1].any():
                continue
            candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def reserve_bbox(
    reserved: np.ndarray, top: int, left: int, obj_h: int, obj_w: int, pad: int = 1
) -> None:
    h, w = reserved.shape
    reserved[
        max(0, top - pad) : min(h, top + obj_h + pad),
        max(0, left - pad) : min(w, left + obj_w + pad),
    ] = True


def draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    h = inner_h + 2
    w = inner_w + 2
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def place_key_within_support(
    grid: np.ndarray,
    key_mask: np.ndarray,
    key_color: int,
    frag_top: int,
    frag_left: int,
    support_mask: np.ndarray,
) -> bool:
    kh, kw = key_mask.shape
    sh, sw = support_mask.shape
    candidates: list[tuple[int, int]] = []
    for rr in range(sh - kh + 1):
        for cc in range(sw - kw + 1):
            if support_mask[rr : rr + kh, cc : cc + kw].all():
                candidates.append((frag_top + rr, frag_left + cc))
    if not candidates:
        return False
    top, left = random.choice(candidates)
    grid[top : top + kh, left : left + kw][key_mask] = key_color
    return True


def gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in (FRAME_COLOR, FRAGMENT_COLOR)])
    n_matched = random.randint(*MATCHED_RANGE)
    n_unmatched = random.randint(*UNMATCHED_RANGE)
    total_fragments = n_matched + n_unmatched

    key_colors = [c for c in range(10) if c not in (bg, FRAME_COLOR, FRAGMENT_COLOR)]
    if len(key_colors) < n_matched:
        return None
    random.shuffle(key_colors)
    key_colors = key_colors[:n_matched]

    used_shapes: set[tuple[int, int, bytes]] = set()
    fragments: list[dict[str, object]] = []
    modes = sample_modes(n_matched)

    for idx in range(total_fragments):
        matched = idx < n_matched
        hole_count = 1 + int(random.random() < 0.6) + int(random.random() < 0.25)
        if not matched:
            hole_count = min(hole_count, 2)

        matched_hole = unique_hole_shape(used_shapes, require_transforms=matched)
        if matched_hole is None:
            return None
        holes = [matched_hole]
        while len(holes) < hole_count:
            extra = unique_hole_shape(used_shapes, require_transforms=False)
            if extra is None:
                return None
            holes.append(extra)
        random.shuffle(holes)

        made = make_fragment(holes)
        if made is None:
            return None
        mask, support = made
        fragments.append(
            {
                "mask": mask,
                "support": support,
                "matched_hole": matched_hole if matched else None,
                "key_color": key_colors[idx] if matched else None,
                "mode": modes[idx] if matched else None,
            }
        )

    frame_specs: list[dict[str, object]] = []
    for idx in range(n_matched):
        frag_mask = fragments[idx]["mask"]  # type: ignore[index]
        support_mask = fragments[idx]["support"]  # type: ignore[index]
        hole_mask = fragments[idx]["matched_hole"]  # type: ignore[index]
        mode = fragments[idx]["mode"]  # type: ignore[index]
        key_mask = choose_key_variant(hole_mask, mode)
        if key_mask is None:
            return None
        fh, fw = frag_mask.shape
        inner_h = fh + random.randint(2, 5)
        inner_w = fw + random.randint(2, 5)
        frame_specs.append(
            {
                "frag_mask": frag_mask,
                "support_mask": support_mask,
                "key_mask": key_mask,
                "key_color": fragments[idx]["key_color"],
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for idx in sorted(
        range(n_matched),
        key=lambda i: -int((frame_specs[i]["inner_h"] + 2) * (frame_specs[i]["inner_w"] + 2)),
    ):
        spec = frame_specs[idx]
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        pos = pick_placement(reserved, frame_h, frame_w, pad=1)
        if pos is None:
            return None
        top, left = pos
        reserve_bbox(reserved, top, left, frame_h, frame_w, pad=1)
        draw_frame(inp, top, left, inner_h, inner_w, FRAME_COLOR)
        draw_frame(out, top, left, inner_h, inner_w, FRAME_COLOR)

        frag_mask = spec["frag_mask"]  # type: ignore[index]
        support_mask = spec["support_mask"]  # type: ignore[index]
        key_mask = spec["key_mask"]  # type: ignore[index]
        fh, fw = frag_mask.shape
        frag_top = top + 1 + (inner_h - fh) // 2
        frag_left = left + 1 + (inner_w - fw) // 2
        if not place_key_within_support(inp, key_mask, int(spec["key_color"]), frag_top, frag_left, support_mask):
            return None
        out[frag_top : frag_top + fh, frag_left : frag_left + fw][frag_mask] = int(spec["key_color"])

    for idx in sorted(
        range(total_fragments),
        key=lambda i: -int(fragments[i]["mask"].shape[0] * fragments[i]["mask"].shape[1]),  # type: ignore[index]
    ):
        frag_mask = fragments[idx]["mask"]  # type: ignore[index]
        fh, fw = frag_mask.shape
        pos = pick_placement(reserved, fh, fw, pad=1)
        if pos is None:
            return None
        top, left = pos
        reserve_bbox(reserved, top, left, fh, fw, pad=1)
        inp[top : top + fh, left : left + fw][frag_mask] = FRAGMENT_COLOR

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
