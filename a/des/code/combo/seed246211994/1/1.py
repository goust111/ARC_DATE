import random
from collections import deque
from math import gcd

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPE_LIBRARY = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
]

HOLE_COUNT_CHOICES = [1, 2]
FRAME_BORDER_CHOICES = [2, 3]
GEN_RETRIES = 400
PACK_RETRIES = 120
SPRITE_PLACE_RETRIES = 160
GRID_MAX_SIDE = 30
GRID_MIN_SIDE = 22


def most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def axis_gap(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return x0 - y1
        if x1 < y0:
            return y0 - x1
        return 0

    return int(axis_gap(ar0, ar1, br0, br1) + axis_gap(ac0, ac1, bc0, bc1))


def component_from_coords(coords: list[tuple[int, int]], color: int | None = None) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return {
        "coords": [(int(r), int(c)) for r, c in coords],
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "size": int(len(coords)),
        "color": None if color is None else int(color),
    }


def component_from_mask(mask: np.ndarray, top: int, left: int, color: int | None = None) -> dict:
    coords = [(int(top + r), int(left + c)) for r, c in np.argwhere(mask)]
    return component_from_coords(coords, color=color)


def find_bool_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(component_from_coords(coords))
    return comps


def find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(component_from_coords(coords, color=color))
    return comps


def extract_frame_holes(grid: np.ndarray, frame: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = frame["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for comp in find_bool_components(sub == bg):
        lr0, lc0, lr1, lc1 = comp["bbox"]
        if lr0 == 0 or lc0 == 0 or lr1 == sub.shape[0] - 1 or lc1 == sub.shape[1] - 1:
            continue
        coords = [(int(r0 + r), int(c0 + c)) for r, c in comp["coords"]]
        holes.append(component_from_coords(coords))
    holes.sort(key=lambda item: item["bbox"])
    return holes


def run_length_gcd(mask: np.ndarray) -> int:
    arr = mask.astype(np.uint8)
    g = 0
    for axis in [arr, arr.T]:
        for line in axis:
            run = 1
            for idx in range(1, len(line)):
                if line[idx] == line[idx - 1]:
                    run += 1
                else:
                    g = gcd(g, int(run))
                    run = 1
            g = gcd(g, int(run))
    return max(1, int(g))


def compress_blocks(mask: np.ndarray, block: int) -> np.ndarray | None:
    h, w = mask.shape
    if block <= 1 or h % block != 0 or w % block != 0:
        return None
    tiles = mask.reshape(h // block, block, w // block, block)
    coarse = tiles[:, 0, :, 0]
    if np.all(tiles == coarse[:, None, :, None]):
        return coarse.copy()
    return None


def normalize_shape(mask: np.ndarray) -> np.ndarray:
    cur = mask.astype(bool)
    while True:
        block = run_length_gcd(cur)
        nxt = compress_blocks(cur, block)
        if nxt is None or nxt.shape == cur.shape:
            return cur
        cur = nxt


def canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    base = normalize_shape(mask)
    best = None
    for rot in range(4):
        turned = np.rot90(base, rot)
        for variant in [turned, np.fliplr(turned)]:
            key = (
                int(variant.shape[0]),
                int(variant.shape[1]),
                variant.astype(np.uint8).tobytes(),
            )
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def fill_hole_rings(out: np.ndarray, hole: dict, color: int, layers: int, bg: int) -> None:
    r0, c0, r1, c1 = hole["bbox"]
    for r, c in hole["coords"]:
        layer = min(r - r0, r1 - r, c - c0, c1 - c)
        out[r, c] = int(color) if layer < int(layers) else int(bg)


def paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    tile = np.ones((factor, factor), dtype=bool)
    return np.kron(mask.astype(np.uint8), tile).astype(bool)


def scale_choices(mask: np.ndarray) -> list[int]:
    area = int(mask.sum())
    choices = []
    for factor in [3, 4, 5]:
        full_min = min(mask.shape) * factor
        max_layers = (full_min + 1) // 2
        if area <= max_layers:
            choices.append(factor)
    return choices or [5]


def build_input_and_output(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    bg = int(spec["bg"])
    frame_color = int(spec["frame_color"])
    h, w = int(spec["grid_shape"][0]), int(spec["grid_shape"][1])
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    fr0, fc0, fr1, fc1 = spec["frame_bbox"]
    inp[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_color
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_color

    for hole in spec["holes"]:
        hr0, hc0 = hole["top"], hole["left"]
        paint_mask(inp, hole["mask"], hr0, hc0, bg)
        fill_hole_rings(out, component_from_mask(hole["mask"], hr0, hc0), hole["color"], hole["layers"], bg)

    for sprite in spec["sprites"]:
        paint_mask(inp, sprite["mask"], sprite["top"], sprite["left"], sprite["color"])

    return inp, out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = most_common_color(g)
    comps = find_monochrome_components(g, bg)
    if not comps:
        return g.copy()

    frame = max(comps, key=lambda comp: comp["size"])
    holes = extract_frame_holes(g, frame, bg)
    sprites_by_key: dict[tuple[int, int, bytes], list[dict]] = {}

    for comp in comps:
        if comp is frame:
            continue
        key = canonical_key(comp["mask"])
        sprites_by_key.setdefault(key, []).append(comp)

    for key in sprites_by_key:
        sprites_by_key[key].sort(key=lambda comp: (comp["size"], comp["bbox"]), reverse=True)

    out = np.full_like(g, bg)
    for r, c in frame["coords"]:
        out[r, c] = int(frame["color"])

    for hole in holes:
        key = canonical_key(hole["mask"])
        matches = sprites_by_key.get(key, [])
        if not matches:
            continue
        sprite = matches.pop(0)
        fill_hole_rings(out, hole, int(sprite["color"]), int(sprite["size"]), bg)

    return out


def sample_hole_specs(hole_count: int) -> list[dict]:
    bases = random.sample(SHAPE_LIBRARY, hole_count)
    colors = random.sample(range(10), hole_count + 2)
    bg = colors[0]
    frame_color = colors[1]
    sprite_colors = colors[2:]

    specs = []
    for base, color in zip(bases, sprite_colors):
        hole_base = random_transform(base)
        sprite_mask = random_transform(base)
        factor = random.choice(scale_choices(base))
        specs.append(
            {
                "hole_mask": scale_mask(hole_base, factor),
                "sprite_mask": sprite_mask,
                "color": int(color),
                "layers": int(sprite_mask.sum()),
            }
        )
    return [{"bg": bg, "frame_color": frame_color, "items": specs}]


def pack_holes(hole_masks: list[np.ndarray], border: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    heights = [int(mask.shape[0]) for mask in hole_masks]
    widths = [int(mask.shape[1]) for mask in hole_masks]
    min_inner_h = max(heights) + 2
    min_inner_w = max(widths) + 2
    max_inner_h = max(min_inner_h, min(22, max(heights) + 9))
    max_inner_w = max(min_inner_w, min(22, max(widths) + 11))

    order = sorted(range(len(hole_masks)), key=lambda idx: hole_masks[idx].sum(), reverse=True)
    for _ in range(PACK_RETRIES):
        inner_h = random.randint(min_inner_h, max_inner_h)
        inner_w = random.randint(min_inner_w, max_inner_w)
        placements = [None] * len(hole_masks)
        bboxes = []
        ok = True
        for idx in order:
            mask = hole_masks[idx]
            h, w = map(int, mask.shape)
            candidates = []
            for r0 in range(inner_h - h + 1):
                for c0 in range(inner_w - w + 1):
                    box = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if all(bbox_gap(box, other) >= 2 for other in bboxes):
                        candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            pos = random.choice(candidates)
            placements[idx] = pos
            bboxes.append((pos[0], pos[1], pos[0] + h - 1, pos[1] + w - 1))
        if ok:
            frame_h = inner_h + 2 * border
            frame_w = inner_w + 2 * border
            if frame_h <= GRID_MAX_SIDE - 4 and frame_w <= GRID_MAX_SIDE - 4:
                return frame_h, frame_w, [(int(r), int(c)) for r, c in placements]
    return None


def place_sprites(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    sprite_masks: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    placements = [None] * len(sprite_masks)
    occupied = [frame_bbox]
    order = sorted(range(len(sprite_masks)), key=lambda idx: sprite_masks[idx].sum(), reverse=True)

    for idx in order:
        mask = sprite_masks[idx]
        sh, sw = map(int, mask.shape)
        candidates = []
        r_start = 1 if h - sh >= 2 else 0
        c_start = 1 if w - sw >= 2 else 0
        r_stop = h - sh if h - sh >= 1 else 0
        c_stop = w - sw if w - sw >= 1 else 0
        for r0 in range(r_start, r_stop + 1):
            for c0 in range(c_start, c_stop + 1):
                box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if all(bbox_gap(box, other) >= 2 for other in occupied):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placements[idx] = pos
        occupied.append((pos[0], pos[1], pos[0] + sh - 1, pos[1] + sw - 1))

    return [(int(r), int(c)) for r, c in placements]


def generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = random.choice(HOLE_COUNT_CHOICES)
    bundle = sample_hole_specs(hole_count)[0]
    bg = int(bundle["bg"])
    frame_color = int(bundle["frame_color"])
    items = bundle["items"]
    border = random.choice(FRAME_BORDER_CHOICES)

    packed = pack_holes([item["hole_mask"] for item in items], border)
    if packed is None:
        return None
    frame_h, frame_w, hole_positions = packed

    min_grid_h = max(GRID_MIN_SIDE, frame_h + 4)
    min_grid_w = max(GRID_MIN_SIDE, frame_w + 4)
    max_grid_h = min(GRID_MAX_SIDE, frame_h + 10)
    max_grid_w = min(GRID_MAX_SIDE, frame_w + 10)
    if min_grid_h > max_grid_h or min_grid_w > max_grid_w:
        return None

    grid_h = random.randint(min_grid_h, max_grid_h)
    grid_w = random.randint(min_grid_w, max_grid_w)
    frame_r0 = random.randint(2, grid_h - frame_h - 2)
    frame_c0 = random.randint(2, grid_w - frame_w - 2)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    sprite_positions = place_sprites(
        (grid_h, grid_w),
        frame_bbox,
        [item["sprite_mask"] for item in items],
    )
    if sprite_positions is None:
        return None

    holes = []
    sprites = []
    for item, (hole_r, hole_c), (sprite_r, sprite_c) in zip(items, hole_positions, sprite_positions):
        holes.append(
            {
                "mask": item["hole_mask"],
                "top": int(frame_r0 + border + hole_r),
                "left": int(frame_c0 + border + hole_c),
                "color": int(item["color"]),
                "layers": int(item["layers"]),
            }
        )
        sprites.append(
            {
                "mask": item["sprite_mask"],
                "top": int(sprite_r),
                "left": int(sprite_c),
                "color": int(item["color"]),
            }
        )

    spec = {
        "bg": bg,
        "frame_color": frame_color,
        "grid_shape": (grid_h, grid_w),
        "frame_bbox": frame_bbox,
        "holes": holes,
        "sprites": sprites,
    }
    inp, expected = build_input_and_output(spec)

    if most_common_color(inp) != bg:
        return None
    comps = find_monochrome_components(inp, bg)
    if len(comps) != 1 + hole_count:
        return None
    if max(comp["size"] for comp in comps) != (frame_h * frame_w - sum(int(hole["mask"].sum()) for hole in holes)):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        attempt = generate_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid sample")
