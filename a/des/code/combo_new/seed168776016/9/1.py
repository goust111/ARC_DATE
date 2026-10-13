from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GEN_RETRIES = 300
KEY_CORNERS = ("tl", "tr", "br", "bl")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(x) for x in mask.shape), mask.tobytes()


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        cur = np.rot90(mask, k)
        key = _mask_key(cur)
        if key in seen:
            continue
        seen.add(key)
        variants.append(cur.copy())
    return variants


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(v) for v in _rotations(mask))


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(int(x) for x in pts[0])
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _analyze_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None
    mask = _mask_from_coords(comp["coords"], comp["bbox"])
    border = _border_mask(h, w)
    if np.any(mask & ~border):
        return None
    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return None
    gap = border & ~mask
    if gap.any() and not _is_connected(gap):
        return None
    if int(mask.sum()) < 2 * (h + w) - 8:
        return None
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "mask": mask,
        "border": border,
        "gap": gap,
        "gap_area": int(gap.sum()),
    }


def _contained_in_frame(comp: dict, frame: dict) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    coords = comp["coords"]
    return bool(
        np.all(coords[:, 0] > r0)
        and np.all(coords[:, 0] < r1)
        and np.all(coords[:, 1] > c0)
        and np.all(coords[:, 1] < c1)
    )


def _shape_mask(comp: dict) -> np.ndarray:
    bbox = comp["bbox"]
    return _mask_from_coords(comp["coords"], bbox)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed key to the lock frame with the same border-notch shape after rotation.

    Input structure:
    - Background is black.
    - Several monochrome hollow rectangular frames are scattered in the grid.
    - Some complete frames contain one small monochrome key shape inside a corner.
    - Other frames contain no inner object but are missing one connected corner-anchored set of border pixels.

    Rule:
    - Identify every rectangular frame and extract either its inner key or its missing border notch.
    - Match each key to the unique notch whose shape equals the key under a 0/90/180/270 degree rotation.
    - Remove the key from its original frame.
    - Repaint the matched lock frame's entire border with the key color, which also fills the missing notch.

    Key insight:
    - The decisive relation is rotation-invariant shape matching between a key component and a frame's missing border segment, followed by color transfer from the key to the completed lock.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_components(g, bg=BG)
    frame_infos = []
    for comp in comps:
        info = _analyze_frame(comp)
        if info is not None:
            frame_infos.append(info)

    frame_by_bbox = {frame["bbox"]: frame for frame in frame_infos}
    inner_components = []
    for comp in comps:
        if comp["bbox"] in frame_by_bbox:
            continue
        hosts = [frame for frame in frame_infos if _contained_in_frame(comp, frame)]
        if len(hosts) == 1:
            inner_components.append({"comp": comp, "frame": hosts[0]})

    keys = []
    for item in inner_components:
        frame = item["frame"]
        if frame["gap_area"] != 0:
            continue
        comp = item["comp"]
        keys.append(
            {
                "frame": frame,
                "comp": comp,
                "color": int(comp["color"]),
                "mask": _shape_mask(comp),
            }
        )

    locks = []
    for frame in frame_infos:
        has_inner = any(item["frame"] is frame for item in inner_components)
        if has_inner:
            continue
        if frame["gap_area"] == 0:
            continue
        gap_mask = _tight_mask(frame["gap"])
        locks.append({"frame": frame, "gap_mask": gap_mask})

    assert keys and locks and len(keys) == len(locks)

    matches = []
    used_locks: set[int] = set()
    for key in keys:
        found = None
        for li, lock in enumerate(locks):
            if li in used_locks:
                continue
            for rot in _rotations(key["mask"]):
                if rot.shape == lock["gap_mask"].shape and np.array_equal(rot, lock["gap_mask"]):
                    if found is not None:
                        raise AssertionError("key matches multiple locks")
                    found = li
                    break
        assert found is not None, "every key must match a lock"
        used_locks.add(found)
        matches.append((key, locks[found]))

    out = g.copy()
    for key, lock in matches:
        out[key["comp"]["coords"][:, 0], key["comp"]["coords"][:, 1]] = BG
        r0, c0, r1, c1 = lock["frame"]["bbox"]
        border = lock["frame"]["border"]
        out[r0 : r1 + 1, c0 : c1 + 1][border] = int(key["color"])

    return out


def _corner_rotation(corner: str) -> int:
    if corner == "tl":
        return 0
    if corner == "tr":
        return 3
    if corner == "br":
        return 2
    if corner == "bl":
        return 1
    raise ValueError(corner)


def _oriented_shape(mask: np.ndarray, corner: str) -> np.ndarray:
    return np.rot90(mask, _corner_rotation(corner))


def _place_bbox(shape: tuple[int, int], outer_shape: tuple[int, int], corner: str, inner: bool) -> tuple[int, int]:
    h, w = shape
    oh, ow = outer_shape
    if inner:
        if corner == "tl":
            return 1, 1
        if corner == "tr":
            return 1, ow - 1 - w
        if corner == "br":
            return oh - 1 - h, ow - 1 - w
        if corner == "bl":
            return oh - 1 - h, 1
    else:
        if corner == "tl":
            return 0, 0
        if corner == "tr":
            return 0, ow - w
        if corner == "br":
            return oh - h, ow - w
        if corner == "bl":
            return oh - h, 0
    raise ValueError(corner)


def _make_key_shape(a: int, b: int) -> np.ndarray:
    mask = np.zeros((b, a), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    return mask


def _full_frame_patch(h: int, w: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _build_source_patch(
    frame_color: int,
    key_color: int,
    key_mask: np.ndarray,
    corner: str,
    outer_h: int,
    outer_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = _full_frame_patch(outer_h, outer_w, frame_color)
    out = inp.copy()
    shape = _oriented_shape(key_mask, corner)
    r0, c0 = _place_bbox(shape.shape, inp.shape, corner, inner=True)
    coords = np.argwhere(shape)
    inp[r0 + coords[:, 0], c0 + coords[:, 1]] = key_color
    return inp, out


def _build_lock_patch(
    frame_color: int,
    key_color: int,
    key_mask: np.ndarray,
    corner: str,
    outer_h: int,
    outer_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = _full_frame_patch(outer_h, outer_w, frame_color)
    out = _full_frame_patch(outer_h, outer_w, key_color)
    shape = _oriented_shape(key_mask, corner)
    r0, c0 = _place_bbox(shape.shape, inp.shape, corner, inner=False)
    coords = np.argwhere(shape)
    inp[r0 + coords[:, 0], c0 + coords[:, 1]] = BG
    return inp, out


def _slot_layout(n_obj: int) -> tuple[int, int]:
    if n_obj <= 2:
        return random.choice([(1, 2), (2, 1)])
    if n_obj <= 4:
        return 2, 2
    return random.choice([(2, 3), (3, 2)])


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([1, 2, 3])
    need_colors = 3 * n_pairs
    colors = random.sample(list(range(1, 10)), need_colors)

    shape_specs = []
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(n_pairs):
        for _try in range(50):
            a = random.randint(2, 4)
            b = random.randint(2, 4)
            shape = _make_key_shape(a, b)
            key = _canonical_rotation_key(shape)
            if key in used_shapes:
                continue
            used_shapes.add(key)
            shape_specs.append(shape)
            break
        else:
            return None

    objects = []
    for i, shape in enumerate(shape_specs):
        src_frame_color = colors[3 * i]
        lock_frame_color = colors[3 * i + 1]
        key_color = colors[3 * i + 2]

        sh, sw = (int(x) for x in shape.shape)
        src_outer_h = sh + random.randint(3, 4)
        src_outer_w = sw + random.randint(3, 4)
        lock_outer_h = sh + random.randint(3, 4)
        lock_outer_w = sw + random.randint(3, 4)

        src_corner = random.choice(KEY_CORNERS)
        lock_corner = random.choice(KEY_CORNERS)

        src_in, src_out = _build_source_patch(
            src_frame_color, key_color, shape, src_corner, src_outer_h, src_outer_w
        )
        lock_in, lock_out = _build_lock_patch(
            lock_frame_color, key_color, shape, lock_corner, lock_outer_h, lock_outer_w
        )
        objects.append({"inp": src_in, "out": src_out})
        objects.append({"inp": lock_in, "out": lock_out})

    random.shuffle(objects)
    n_obj = len(objects)
    rows, cols = _slot_layout(n_obj)
    if rows * cols < n_obj:
        return None

    max_h = max(int(obj["inp"].shape[0]) for obj in objects)
    max_w = max(int(obj["inp"].shape[1]) for obj in objects)
    cell_h = max_h + 3
    cell_w = max_w + 3
    need_h = rows * cell_h + 1
    need_w = cols * cell_w + 1
    if need_h > 30 or need_w > 30:
        return None

    H = random.randint(need_h, min(30, need_h + 4))
    W = random.randint(need_w, min(30, need_w + 4))
    slot_h = H // rows
    slot_w = W // cols
    if slot_h < max_h + 2 or slot_w < max_w + 2:
        return None

    inp = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    slot_ids = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(slot_ids)
    for obj, (sr, sc) in zip(objects, slot_ids):
        patch_in = obj["inp"]
        patch_out = obj["out"]
        ph, pw = patch_in.shape
        top_min = sr * slot_h + 1
        left_min = sc * slot_w + 1
        top_max = min((sr + 1) * slot_h - ph - 1, H - ph)
        left_max = min((sc + 1) * slot_w - pw - 1, W - pw)
        if top_min > top_max or left_min > left_max:
            return None
        r0 = random.randint(top_min, top_max)
        c0 = random.randint(left_min, left_max)
        inp[r0 : r0 + ph, c0 : c0 + pw] = patch_in
        out[r0 : r0 + ph, c0 : c0 + pw] = patch_out

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
