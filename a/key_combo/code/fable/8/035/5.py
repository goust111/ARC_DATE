from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox, crop_bbox, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, b):
        r0, c0, r1, c1 = map(int, b)
        return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()
USED_SIGNATURES: set[bytes] = set()


def _classify_block(block: np.ndarray, bg: int) -> dict:
    mask = block != bg
    b = bbox(mask)
    if b is None:
        return {"kind": "empty"}

    r0, c0, r1, c1 = map(int, b)
    sprite = crop_bbox(block, b)
    colors = [int(c) for c in np.unique(sprite) if int(c) != bg]
    full = bool(np.all(sprite != bg))
    mask_local = np.zeros(block.shape, dtype=bool)
    mask_local[r0 : r1 + 1, c0 : c1 + 1] = block[r0 : r1 + 1, c0 : c1 + 1] != bg

    if len(colors) == 1 and full:
        return {
            "kind": "rect",
            "color": int(colors[0]),
            "bbox": (r0, c0, r1, c1),
            "mask": mask_local,
            "sprite": sprite,
        }
    return {
        "kind": "other",
        "bbox": (r0, c0, r1, c1),
        "mask": mask_local,
        "sprite": sprite,
        "colors": colors,
    }


def _analyze_candidate(grid: np.ndarray, h: int, w: int, bg: int) -> dict | None:
    gh, gw = map(int, grid.shape)
    if gh % h != 0 or gw % w != 0:
        return None
    block_h, block_w = gh // h, gw // w
    if block_h < h or block_w < w:
        return None

    rects: dict[tuple[int, int], dict] = {}
    map_candidates: list[tuple[int, int, dict]] = []
    seen_rect_colors: set[int] = set()

    for br in range(h):
        for bc in range(w):
            block = grid[br * block_h : (br + 1) * block_h, bc * block_w : (bc + 1) * block_w]
            info = _classify_block(block, bg)
            if info["kind"] == "empty":
                continue
            if info["kind"] == "rect":
                color = int(info["color"])
                if color in seen_rect_colors:
                    return None
                seen_rect_colors.add(color)
                rects[(br, bc)] = info
                continue
            sprite = np.asarray(info["sprite"], dtype=int)
            colors = [int(c) for c in info["colors"]]
            if sprite.shape == (h, w) and len(colors) >= 2:
                map_candidates.append((br, bc, info))
                continue
            return None

    if len(rects) < 3 or len(map_candidates) != 1:
        return None

    map_r, map_c, map_info = map_candidates[0]
    micro = np.asarray(map_info["sprite"], dtype=int)
    if int(micro[map_r, map_c]) != bg:
        return None

    live_cells = [(r, c) for r in range(h) for c in range(w) if int(micro[r, c]) != bg]
    if len(live_cells) < 3:
        return None
    if len({int(micro[r, c]) for r, c in live_cells}) < 2:
        return None

    recolor_count = 0
    erase_count = 0
    visible_count = 0
    for r in range(h):
        for c in range(w):
            if (r, c) == (map_r, map_c):
                continue
            instr = int(micro[r, c])
            has_rect = (r, c) in rects
            if instr != bg:
                visible_count += 1
                if not has_rect:
                    return None
                if int(rects[(r, c)]["color"]) != instr:
                    recolor_count += 1
            elif has_rect:
                erase_count += 1

    if visible_count < 3 or recolor_count + erase_count == 0:
        return None

    return {
        "h": int(h),
        "w": int(w),
        "block_h": int(block_h),
        "block_w": int(block_w),
        "map_pos": (int(map_r), int(map_c)),
        "micro": micro,
        "rects": rects,
        "recolor_count": int(recolor_count),
        "erase_count": int(erase_count),
    }


def _find_candidate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    gh, gw = map(int, g.shape)

    candidates = []
    for h in range(2, gh + 1):
        if gh % h != 0:
            continue
        for w in range(2, gw + 1):
            if gw % w != 0:
                continue
            cand = _analyze_candidate(g, h, w, bg)
            if cand is not None:
                candidates.append(cand)

    assert len(candidates) == 1, f"expected one valid partition candidate, got {len(candidates)}"
    cand = candidates[0]
    cand["bg"] = bg
    return cand


def _render_output_from_candidate(grid: np.ndarray, cand: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(cand["bg"])
    out = np.full_like(g, bg)
    micro = np.asarray(cand["micro"], dtype=int)
    block_h = int(cand["block_h"])
    block_w = int(cand["block_w"])

    for (br, bc), info in cand["rects"].items():
        instr = int(micro[br, bc])
        if instr == bg:
            continue
        r0 = int(br) * block_h
        c0 = int(bc) * block_w
        view = out[r0 : r0 + block_h, c0 : c0 + block_w]
        view[np.asarray(info["mask"], dtype=bool)] = instr
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the hidden micro-map block as per-cell instructions for recoloring or erasing the larger block rectangles.

    Input structure:
    - The canvas splits evenly into an h by w lattice of equal-sized blocks on a black background.
    - Most non-empty blocks contain exactly one monochrome solid rectangle, with at most one such rectangle per block; some blocks are empty.
    - One special block instead contains a multicolor h by w miniature whose own bounding box size reveals the lattice dimensions, and whose cell at its own block coordinate is black.

    Rule:
    - Find the unique valid partition where exactly one block contains that multicolor h by w miniature and every other non-empty block is a solid rectangle.
    - Read miniature cell (i, j) as the instruction for block (i, j): a non-black color recolors that block's rectangle to the instruction color, while black clears any rectangle in that block.
    - The miniature block is itself removed in the output because it is only an instruction object.

    Key insight:
    - A normal-looking small object is actually a downscaled control map: its size determines the block grid, and its cell colors deterministically drive the blockwise edits.
    """
    cand = _find_candidate(np.asarray(grid, dtype=int))
    return _render_output_from_candidate(np.asarray(grid, dtype=int), cand)


def _choose_geometry() -> tuple[int, int, int, int] | None:
    shapes = [(h, w) for h in range(2, 6) for w in range(2, 6) if h * w >= 6]
    RNG.shuffle(shapes)
    for h, w in shapes:
        bh_lo = max(h, 3)
        bw_lo = max(w, 3)
        bh_hi = 30 // h
        bw_hi = 30 // w
        if bh_lo > bh_hi or bw_lo > bw_hi:
            continue
        bh = int(RNG.integers(bh_lo, bh_hi + 1))
        bw = int(RNG.integers(bw_lo, bw_hi + 1))
        if h * bh <= 30 and w * bw <= 30:
            return int(h), int(w), int(bh), int(bw)
    return None


def _sample_survivors(cells: list[tuple[int, int]], h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    remaining = cells.copy()
    RNG.shuffle(remaining)
    chosen: list[tuple[int, int]] = []
    constraints = {
        "top": lambda p: p[0] == 0,
        "bottom": lambda p: p[0] == h - 1,
        "left": lambda p: p[1] == 0,
        "right": lambda p: p[1] == w - 1,
    }
    covered = {key: False for key in constraints}

    for key, pred in constraints.items():
        options = [p for p in remaining if pred(p)]
        if not options:
            return None
        pick = options[int(RNG.integers(len(options)))]
        if pick not in chosen:
            chosen.append(pick)
        covered[key] = True

    extras = [p for p in remaining if p not in chosen]
    RNG.shuffle(extras)
    for p in extras:
        if len(chosen) >= count:
            break
        chosen.append(p)

    if len(chosen) < count:
        return None
    return chosen[:count]


def _draw_rect(block_h: int, block_w: int) -> np.ndarray:
    rect_h_lo = 2 if block_h >= 4 else 1
    rect_w_lo = 2 if block_w >= 4 else 1
    rect_h = int(RNG.integers(rect_h_lo, block_h + 1))
    rect_w = int(RNG.integers(rect_w_lo, block_w + 1))
    r0 = int(RNG.integers(0, block_h - rect_h + 1))
    c0 = int(RNG.integers(0, block_w - rect_w + 1))
    mask = np.zeros((block_h, block_w), dtype=bool)
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
    return mask


def _make_instruction_colors(source_by_pos: dict[tuple[int, int], int], survivors: list[tuple[int, int]]) -> dict[tuple[int, int], int] | None:
    distinct_target_count = min(len(survivors), int(RNG.integers(2, min(5, len(survivors)) + 1)))
    target_palette = list(RNG.choice(PALETTE, size=distinct_target_count, replace=False))
    instructions: dict[tuple[int, int], int] = {}

    shuffled = survivors.copy()
    RNG.shuffle(shuffled)
    groups = [target_palette[i % distinct_target_count] for i in range(len(shuffled))]
    RNG.shuffle(groups)
    for pos, color in zip(shuffled, groups):
        if int(source_by_pos[pos]) == int(color):
            alt = [c for c in PALETTE if int(c) != int(source_by_pos[pos])]
            if not alt:
                return None
            color = int(alt[int(RNG.integers(len(alt)))])
        instructions[pos] = int(color)

    if len(set(instructions.values())) < 2:
        return None
    return instructions


def _build_scene() -> tuple[np.ndarray, np.ndarray, dict] | None:
    geom = _choose_geometry()
    if geom is None:
        return None
    h, w, block_h, block_w = geom
    gh, gw = h * block_h, w * block_w

    all_cells = [(r, c) for r in range(h) for c in range(w)]
    map_pos = all_cells[int(RNG.integers(len(all_cells)))]
    usable = [p for p in all_cells if p != map_pos]
    if len(usable) < 4:
        return None

    max_occ = min(8, len(usable))
    if max_occ < 4:
        return None
    occ_count = int(RNG.integers(4, max_occ + 1))
    erase_count = int(RNG.integers(1, min(3, occ_count - 1) + 1))
    survive_count = occ_count - erase_count
    if survive_count < 3:
        survive_count = 3
        erase_count = occ_count - survive_count
    if erase_count < 1:
        return None

    survivors = _sample_survivors(usable, h, w, survive_count)
    if survivors is None:
        return None
    remaining = [p for p in usable if p not in survivors]
    RNG.shuffle(remaining)
    erasers = remaining[:erase_count]
    occupied = survivors + erasers
    if len(occupied) != occ_count:
        return None
    if len(remaining) == erase_count:
        return None

    source_colors = list(RNG.choice(PALETTE, size=occ_count, replace=False))
    source_by_pos = {pos: int(color) for pos, color in zip(occupied, source_colors)}
    instruction_colors = _make_instruction_colors(source_by_pos, survivors)
    if instruction_colors is None:
        return None

    micro = np.zeros((h, w), dtype=int)
    for pos, color in instruction_colors.items():
        micro[pos] = int(color)
    if int(micro[map_pos]) != BG:
        return None
    if micro.shape != (h, w):
        return None
    if len({int(c) for c in np.unique(micro) if int(c) != BG}) < 2:
        return None
    if bbox(micro != BG) != (0, 0, h - 1, w - 1):
        return None

    inp = np.zeros((gh, gw), dtype=int)
    out = np.zeros_like(inp)

    rect_masks: dict[tuple[int, int], np.ndarray] = {}
    for pos in occupied:
        rect_masks[pos] = _draw_rect(block_h, block_w)

    for (br, bc), color in source_by_pos.items():
        r0 = br * block_h
        c0 = bc * block_w
        mask = rect_masks[(br, bc)]
        inp[r0 : r0 + block_h, c0 : c0 + block_w][mask] = int(color)

    map_top = int(RNG.integers(0, block_h - h + 1))
    map_left = int(RNG.integers(0, block_w - w + 1))
    mr, mc = map_pos
    map_r0 = mr * block_h
    map_c0 = mc * block_w
    inp[map_r0 + map_top : map_r0 + map_top + h, map_c0 + map_left : map_c0 + map_left + w] = micro

    for (br, bc), color in instruction_colors.items():
        r0 = br * block_h
        c0 = bc * block_w
        out[r0 : r0 + block_h, c0 : c0 + block_w][rect_masks[(br, bc)]] = int(color)

    meta = {
        "h": h,
        "w": w,
        "block_h": block_h,
        "block_w": block_w,
        "map_pos": map_pos,
        "source_by_pos": source_by_pos,
        "instruction_colors": instruction_colors,
        "erasers": set(erasers),
    }
    return inp, out, meta


def _semantic_check(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        cand = _find_candidate(inp)
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if (int(cand["h"]), int(cand["w"])) != (int(meta["h"]), int(meta["w"])):
        return False
    if tuple(map(int, cand["map_pos"])) != tuple(map(int, meta["map_pos"])):
        return False
    if int(cand["erase_count"]) < 1 or int(cand["recolor_count"]) < 1:
        return False
    if len(meta["instruction_colors"]) < 3 or len(meta["erasers"]) < 1:
        return False
    if len({int(c) for c in meta["instruction_colors"].values()}) < 2:
        return False
    if not any(pos in meta["source_by_pos"] for pos in meta["erasers"]):
        return False
    if np.array_equal(inp, expected):
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    for pos, source_color in meta["source_by_pos"].items():
        br, bc = pos
        r0 = br * meta["block_h"]
        c0 = bc * meta["block_w"]
        block_in = inp[r0 : r0 + meta["block_h"], c0 : c0 + meta["block_w"]]
        block_out = actual[r0 : r0 + meta["block_h"], c0 : c0 + meta["block_w"]]
        target_color = int(meta["instruction_colors"].get(pos, BG))
        if pos in meta["erasers"]:
            if np.any(block_out != BG):
                return False
        elif target_color != BG:
            if np.count_nonzero(block_in == int(source_color)) == 0:
                return False
            if np.count_nonzero(block_out == target_color) == 0:
                return False
            if np.any((block_out != BG) & (block_out != target_color)):
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(800):
        built = _build_scene()
        if built is None:
            continue
        inp, expected, meta = built
        if not _semantic_check(inp, expected, meta):
            continue
        sig = inp.tobytes() + str(inp.shape).encode() + expected.tobytes()
        if sig in USED_SIGNATURES:
            continue
        USED_SIGNATURES.add(sig)
        return inp
    raise RuntimeError("failed to generate a valid micro-map self-instruction sample")
