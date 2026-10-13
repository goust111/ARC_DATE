import random
from collections import defaultdict

import numpy as np


BG = 0
GRID_LIMIT = 30
GEN_RETRIES = 400
PAIR_RETRIES = 200
TEMPLATE_RETRIES = 200
PLACEMENT_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _hole_infos(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if touches_border:
                continue
            coords = np.asarray(cells, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            rel = coords - np.asarray([[r0, c0]])
            hole_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            hole_mask[rel[:, 0], rel[:, 1]] = True
            holes.append({"coords": coords, "bbox": (r0, c0, r1, c1), "mask": hole_mask})
    return holes


def _filled_mask(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    holes = _hole_infos(mask)
    filled = np.array(mask, dtype=bool, copy=True)
    for hole in holes:
        filled[hole["coords"][:, 0], hole["coords"][:, 1]] = True
    return filled, holes


def _canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best = None
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _resize_nearest(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = map(int, mask.shape)
    row_idx = ((2 * np.arange(target_h) + 1) * src_h) // (2 * target_h)
    col_idx = ((2 * np.arange(target_w) + 1) * src_w) // (2 * target_w)
    return mask[row_idx[:, None], col_idx[None, :]]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = _components(grid != BG)
    objects: list[dict] = []
    for coords in comps:
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1 and int(colors[0]) != BG
        color = int(colors[0])
        r0, c0, r1, c1 = _bbox(coords)
        rel = coords - np.asarray([[r0, c0]])
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        filled, holes = _filled_mask(mask)
        objects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "filled_mask": filled,
                "holes": holes,
                "color": color,
            }
        )
    return objects


def _apply_pair(out: np.ndarray, container: dict, content: dict) -> None:
    assert len(container["holes"]) == 1
    hole = container["holes"][0]
    hole_mask = hole["mask"]
    assert bool(hole_mask.all())
    hr0, hc0, hr1, hc1 = hole["bbox"]
    hole_h = int(hr1 - hr0 + 1)
    hole_w = int(hc1 - hc0 + 1)

    scaled = _resize_nearest(content["mask"], hole_h, hole_w)
    patch = np.full((hole_h, hole_w), BG, dtype=int)
    patch[scaled] = int(content["color"])

    out[content["coords"][:, 0], content["coords"][:, 1]] = BG
    ar0 = int(container["bbox"][0] + hr0)
    ac0 = int(container["bbox"][1] + hc0)
    out[ar0 : ar0 + hole_h, ac0 : ac0 + hole_w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each hollow object with its matching solid silhouette and stamp a scaled copy of the solid pattern into the hollow cavity.

    Input structure:
    - The background is black.
    - The foreground consists only of separated monochrome 4-connected objects.
    - Every object belongs to exactly one shape pair: the two members share the same filled silhouette up to 90-degree rotation.
    - In each pair, one member has one enclosed rectangular black hole, while the other has no enclosed hole.

    Rule:
    - Extract every object and compare their hole-filled silhouettes under rotation to recover the pairs.
    - In each pair, treat the holed member as the container and the hole-free member as the content.
    - Crop the content to its own bounding box, resize that binary color pattern to the container hole size with nearest-neighbor scaling, write it into the rectangular hole using the content color, and erase the original content object to black.

    Key insight:
    - The pairing cue is the shared outer silhouette, not the raw pixels: the container only differs by an internal rectangular cavity, and that cavity receives a scaled-down copy of the content object's own shape pattern rather than a flat fill.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    assert objects

    groups: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    for obj in objects:
        groups[_canonical_key(obj["filled_mask"])].append(obj)

    out = g.copy()
    for group in groups.values():
        assert len(group) == 2
        containers = [obj for obj in group if obj["holes"]]
        contents = [obj for obj in group if not obj["holes"]]
        assert len(containers) == 1 and len(contents) == 1
        _apply_pair(out, containers[0], contents[0])

    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _random_template(h: int, w: int) -> np.ndarray | None:
    for _ in range(TEMPLATE_RETRIES):
        mask = np.ones((h, w), dtype=bool)
        n_notches = random.randint(1, min(4, h + w - 4))
        for _ in range(n_notches):
            edge = random.choice(["top", "bottom", "left", "right"])
            if edge in {"top", "bottom"}:
                if w <= 2:
                    continue
                span = random.randint(1, w - 1)
                c0 = random.randint(0, w - span)
                r = 0 if edge == "top" else h - 1
                mask[r, c0 : c0 + span] = False
            else:
                if h <= 2:
                    continue
                span = random.randint(1, h - 1)
                r0 = random.randint(0, h - span)
                c = 0 if edge == "left" else w - 1
                mask[r0 : r0 + span, c] = False
        if bool(mask.all()):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if int(mask.sum()) < max(5, int(0.6 * h * w)):
            continue
        if not _is_connected(mask):
            continue
        if _hole_infos(mask):
            continue
        return mask
    return None


def _cavity_positions(mask: np.ndarray, hole_h: int, hole_w: int) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    out: list[tuple[int, int]] = []
    for r in range(1, h - hole_h):
        for c in range(1, w - hole_w):
            ring = mask[r - 1 : r + hole_h + 1, c - 1 : c + hole_w + 1]
            if ring.shape != (hole_h + 2, hole_w + 2):
                continue
            if bool(ring.all()):
                out.append((r, c))
    return out


def _build_pair(used_keys: set[tuple[int, int, bytes]], small_mode: bool) -> dict | None:
    for _ in range(PAIR_RETRIES):
        hole_h = random.randint(3, 4 if small_mode else 5)
        if random.random() < 0.4:
            hole_w = hole_h
        else:
            hole_w = random.randint(3, 5 if small_mode else 6)
        scale = random.choice([2, 2, 3] if small_mode else [2, 3])

        template = _random_template(hole_h, hole_w)
        if template is None:
            continue
        base = np.kron(template.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)
        key = _canonical_key(base)
        if key in used_keys:
            continue

        content_rot = random.randrange(4)
        content_mask = np.rot90(base, content_rot)
        target_h = int(content_mask.shape[0] // scale)
        target_w = int(content_mask.shape[1] // scale)

        rots = list(range(4))
        random.shuffle(rots)
        chosen = None
        for container_rot in rots:
            container_filled = np.rot90(base, container_rot)
            positions = _cavity_positions(container_filled, target_h, target_w)
            if not positions:
                continue
            hole_r, hole_c = random.choice(positions)
            container_mask = container_filled.copy()
            container_mask[hole_r : hole_r + target_h, hole_c : hole_c + target_w] = False
            if not _is_connected(container_mask):
                continue
            holes = _hole_infos(container_mask)
            if len(holes) != 1:
                continue
            if holes[0]["bbox"] != (hole_r, hole_c, hole_r + target_h - 1, hole_c + target_w - 1):
                continue
            if not bool(holes[0]["mask"].all()):
                continue
            chosen = {
                "key": key,
                "content_mask": content_mask,
                "container_mask": container_mask,
                "hole_bbox": holes[0]["bbox"],
            }
            break
        if chosen is None:
            continue
        return chosen
    return None


def _boxes_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_objects(objects: list[dict]) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int, int, int]] | None:
    canvas = np.zeros((GRID_LIMIT, GRID_LIMIT), dtype=int)
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    boxes: list[tuple[int, int, int, int]] = []

    order = sorted(
        range(len(objects)),
        key=lambda idx: int(objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1]),
        reverse=True,
    )

    for idx in order:
        mask = objects[idx]["mask"]
        h, w = map(int, mask.shape)
        if h > GRID_LIMIT - 2 or w > GRID_LIMIT - 2:
            return None

        candidates: list[tuple[int, int]] = []
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(1, GRID_LIMIT - h - 1)
            c = random.randint(1, GRID_LIMIT - w - 1)
            box = (r, c, r + h - 1, c + w - 1)
            if all(_boxes_apart(box, prev, gap=1) for prev in boxes):
                candidates.append((r, c))
                if len(candidates) >= 20:
                    break
        if not candidates:
            for r in range(1, GRID_LIMIT - h):
                for c in range(1, GRID_LIMIT - w):
                    box = (r, c, r + h - 1, c + w - 1)
                    if all(_boxes_apart(box, prev, gap=1) for prev in boxes):
                        candidates.append((r, c))
            if not candidates:
                return None

        pos = random.choice(candidates)
        placements[idx] = pos
        boxes.append((pos[0], pos[1], pos[0] + h - 1, pos[1] + w - 1))

    for idx, obj in enumerate(objects):
        r, c = placements[idx]
        patch = np.full(obj["mask"].shape, BG, dtype=int)
        patch[obj["mask"]] = int(obj["color"])
        region = canvas[r : r + patch.shape[0], c : c + patch.shape[1]]
        region[patch != BG] = patch[patch != BG]

    fg = np.argwhere(canvas != BG)
    assert len(fg) > 0
    r0, c0, r1, c1 = _bbox(fg)
    top = random.randint(1, min(4, r0))
    left = random.randint(1, min(4, c0))
    bottom = random.randint(1, min(4, GRID_LIMIT - 1 - r1))
    right = random.randint(1, min(4, GRID_LIMIT - 1 - c1))
    crop_box = (r0 - top, c0 - left, r1 + bottom, c1 + right)
    cr0, cc0, cr1, cc1 = crop_box
    cropped = canvas[cr0 : cr1 + 1, cc0 : cc1 + 1].copy()
    final_positions = [(r - cr0, c - cc0) for r, c in placements]
    return cropped, final_positions, crop_box


def _expected_output(inp: np.ndarray, objects: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    by_pair: dict[int, dict] = defaultdict(dict)
    for obj, (r, c) in zip(objects, positions):
        by_pair[int(obj["pair"])][obj["role"]] = (obj, r, c)

    for pair in by_pair.values():
        container_obj, cr, cc = pair["container"]
        content_obj, ir, ic = pair["content"]

        content_coords = np.argwhere(content_obj["mask"])
        out[ir + content_coords[:, 0], ic + content_coords[:, 1]] = BG

        hr0, hc0, hr1, hc1 = container_obj["hole_bbox"]
        hole_h = int(hr1 - hr0 + 1)
        hole_w = int(hc1 - hc0 + 1)
        scaled = _resize_nearest(content_obj["mask"], hole_h, hole_w)
        patch = np.full((hole_h, hole_w), BG, dtype=int)
        patch[scaled] = int(content_obj["color"])
        out[cr + hr0 : cr + hr0 + hole_h, cc + hc0 : cc + hc0 + hole_w] = patch

    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([2, 2, 3])
    used_keys: set[tuple[int, int, bytes]] = set()
    pairs: list[dict] = []
    for _ in range(n_pairs):
        pair = _build_pair(used_keys, small_mode=n_pairs == 3)
        if pair is None:
            return None
        used_keys.add(pair["key"])
        pairs.append(pair)

    colors = random.sample(list(range(1, 10)), 2 * n_pairs)
    objects: list[dict] = []
    for pair_idx, pair in enumerate(pairs):
        container_color = colors[2 * pair_idx]
        content_color = colors[2 * pair_idx + 1]
        objects.append(
            {
                "pair": pair_idx,
                "role": "container",
                "mask": pair["container_mask"],
                "color": container_color,
                "hole_bbox": pair["hole_bbox"],
            }
        )
        objects.append(
            {
                "pair": pair_idx,
                "role": "content",
                "mask": pair["content_mask"],
                "color": content_color,
            }
        )

    placed = _place_objects(objects)
    if placed is None:
        return None
    inp, positions, _crop_box = placed
    expected = _expected_output(inp, objects, positions)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
