from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BLOCK = 3
GAP = 1
STEP = BLOCK + GAP
MASTER_SLOTS = 7
MASTER_SIZE = 1 + MASTER_SLOTS * STEP
MASTER_SHAPE = (MASTER_SIZE, MASTER_SIZE)
GREY = Color.GREY
BG = Color.BLACK
COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_GENERATION_ATTEMPTS = 400


def _mask_key(mask: np.ndarray) -> str:
    return "".join("1" if cell else "0" for cell in mask.astype(bool).ravel())


def _canonicalize_mask(mask: np.ndarray) -> tuple[str, np.ndarray]:
    rotations = [np.rot90(mask, k) for k in range(4)]
    keys = [_mask_key(rot) for rot in rotations]
    best_idx = min(range(4), key=lambda idx: keys[idx])
    return keys[best_idx], rotations[best_idx]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BLOCK and 0 <= nc < BLOCK and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _has_full_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (int(r0), int(c0), int(r1), int(c1)) == (0, 0, BLOCK - 1, BLOCK - 1)


def _build_symbol_library() -> list[np.ndarray]:
    classes: dict[str, np.ndarray] = {}
    for bits in range(1, 1 << (BLOCK * BLOCK)):
        mask = np.array([(bits >> idx) & 1 for idx in range(BLOCK * BLOCK)], dtype=bool).reshape(BLOCK, BLOCK)
        if not _is_connected(mask):
            continue
        if not _has_full_bbox(mask):
            continue
        if any(np.array_equal(mask, np.rot90(mask, turns)) for turns in (1, 2, 3)):
            continue
        key, canon = _canonicalize_mask(mask)
        area = int(canon.sum())
        if area < 5 or area > 6:
            continue
        classes.setdefault(key, canon.copy())
    library = [classes[key] for key in sorted(classes)]
    assert len(library) >= 8
    return library


SYMBOL_LIBRARY = _build_symbol_library()


def _render_symbol(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros((BLOCK, BLOCK), dtype=int)
    patch[mask.astype(bool)] = GREY
    return patch


def _render_square(color: int) -> np.ndarray:
    return np.full((BLOCK, BLOCK), int(color), dtype=int)


def _slot_to_rc(slot: tuple[int, int]) -> tuple[int, int]:
    sr, sc = slot
    return 1 + sr * STEP, 1 + sc * STEP


def _component_mask(component: dict, grid: np.ndarray) -> np.ndarray:
    patch = crop_bbox(grid, component["bbox"])
    return patch == GREY


def _bbox_shape(bbox_: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _parse_grid(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    symbols: list[dict] = []
    blocks: dict[tuple[int, int], dict] = {}
    for component in components:
        bbox_ = component["bbox"]
        h, w = _bbox_shape(bbox_)
        assert (h, w) == (BLOCK, BLOCK), f"every foreground block must occupy a 3x3 bbox, got {bbox_}"
        r0, c0, _r1, _c1 = bbox_
        tl = (int(r0), int(c0))
        color = int(component["color"])
        if color == GREY:
            mask = _component_mask(component, g)
            assert 0 < int(mask.sum()) < BLOCK * BLOCK, "glyph mask cannot be empty or solid"
            key, _canon = _canonicalize_mask(mask)
            record = {
                "bbox": bbox_,
                "tl": tl,
                "mask": mask,
                "key": key,
            }
            symbols.append(record)
            blocks[tl] = {"kind": "symbol", **record}
        else:
            assert int(component["area"]) == BLOCK * BLOCK, "colored blocks must be solid 3x3 squares"
            blocks[tl] = {"kind": "color", "bbox": bbox_, "tl": tl, "color": color}

    assert symbols, "expected at least one glyph"
    return symbols, blocks


def _resolve_color(key: str, direct: dict[str, int], synonym: dict[str, str]) -> int:
    seen = set()
    cur = key
    while cur not in direct:
        assert cur in synonym, f"unresolved glyph {cur}"
        assert cur not in seen, "synonym chain must be acyclic"
        seen.add(cur)
        cur = synonym[cur]
    return int(direct[cur])


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decode isolated gray glyph blocks by rotation-equivalent in-grid legend pairs, then replace them with solid color blocks.

    Input structure:
    - Black background with non-overlapping 3x3 foreground blocks.
    - Gray 3x3 glyph masks appear either as legend entries or as isolated work items.
    - A legend pair is a gray glyph whose right-neighbor block is either another gray glyph
      (synonym jump) or a solid colored 3x3 square (direct color definition).

    Rule:
    - Build direct glyph-to-color mappings from glyph->color pairs and glyph-to-glyph jumps from
      glyph->glyph pairs, matching glyph identities up to 90-degree rotations.
    - For each isolated glyph block with no left or right neighbor block, follow its synonym chain
      until a direct color definition is reached.
    - Erase every legend pair and replace each isolated glyph's full 3x3 bbox with a solid square
      of the resolved color.

    Key insight:
    - The codebook is defined locally by horizontal adjacency inside the same grid, while glyph
      identity itself is invariant under rotation; solving requires combining both signals.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    symbols, blocks = _parse_grid(g)
    direct: dict[str, int] = {}
    synonym: dict[str, str] = {}
    work_items: list[dict] = []

    for symbol in symbols:
        r0, c0 = symbol["tl"]
        left_tl = (r0, c0 - STEP)
        right_tl = (r0, c0 + STEP)
        has_left = left_tl in blocks
        right = blocks.get(right_tl)
        if right is not None:
            assert not has_left, "glyph cannot be both a pair source and a pair target"
            if right["kind"] == "color":
                direct[symbol["key"]] = int(right["color"])
            else:
                synonym[symbol["key"]] = right["key"]
        elif not has_left:
            work_items.append(symbol)

    assert direct, "expected at least one direct definition pair"
    assert work_items, "expected isolated glyphs to decode"

    out = np.full_like(g, int(bg))
    for symbol in work_items:
        color = _resolve_color(symbol["key"], direct, synonym)
        r0, c0, r1, c1 = symbol["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _resolve_symbol_index(symbol_idx: int, direct_colors: dict[int, int], synonym_next: dict[int, int]) -> int:
    seen = set()
    cur = int(symbol_idx)
    while cur not in direct_colors:
        assert cur in synonym_next
        assert cur not in seen
        seen.add(cur)
        cur = synonym_next[cur]
    return int(direct_colors[cur])


def _choose_relations(symbol_count: int) -> tuple[list[int], dict[int, int], dict[int, int]]:
    order = random.sample(range(symbol_count), symbol_count)
    direct_count = random.randint(2, min(3, symbol_count - 2))
    direct_nodes = set(order[-direct_count:])
    direct_colors = {
        node: color
        for node, color in zip(sorted(direct_nodes), random.sample(COLOR_POOL, direct_count))
    }
    synonym_next: dict[int, int] = {}
    synonym_nodes = order[:-direct_count]
    for idx, node in enumerate(synonym_nodes):
        later = order[idx + 1 :]
        if idx == 0 and len(synonym_nodes) >= 2:
            synonym_next[node] = synonym_nodes[1]
        else:
            synonym_next[node] = random.choice(later)
    return order, direct_colors, synonym_next


def _find_placement(pair_count: int, work_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(300):
        occupied: set[tuple[int, int]] = set()
        pair_slots: list[tuple[int, int]] = []
        ok = True
        for _pair_idx in range(pair_count):
            candidates = []
            for sr in range(MASTER_SLOTS):
                for sc in range(MASTER_SLOTS - 1):
                    left = (sr, sc)
                    right = (sr, sc + 1)
                    if left in occupied or right in occupied:
                        continue
                    if (sr, sc - 1) in occupied or (sr, sc + 2) in occupied:
                        continue
                    candidates.append(left)
            if not candidates:
                ok = False
                break
            left = random.choice(candidates)
            sr, sc = left
            occupied.add((sr, sc))
            occupied.add((sr, sc + 1))
            pair_slots.append(left)
        if not ok:
            continue

        work_slots: list[tuple[int, int]] = []
        for _work_idx in range(work_count):
            candidates = []
            for sr in range(MASTER_SLOTS):
                for sc in range(MASTER_SLOTS):
                    slot = (sr, sc)
                    if slot in occupied:
                        continue
                    if (sr, sc - 1) in occupied or (sr, sc + 1) in occupied:
                        continue
                    candidates.append(slot)
            if not candidates:
                ok = False
                break
            slot = random.choice(candidates)
            occupied.add(slot)
            work_slots.append(slot)
        if ok:
            random.shuffle(pair_slots)
            random.shuffle(work_slots)
            return pair_slots, work_slots
    return None


def _crop_canvas(canvas: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(canvas != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_pad = random.randint(0, min(2, int(r0)))
    left_pad = random.randint(0, min(2, int(c0)))
    bottom_pad = random.randint(0, min(2, canvas.shape[0] - 1 - int(r1)))
    right_pad = random.randint(0, min(2, canvas.shape[1] - 1 - int(c1)))
    crop_r0 = int(r0) - top_pad
    crop_c0 = int(c0) - left_pad
    cropped = canvas[crop_r0 : int(r1) + bottom_pad + 1, crop_c0 : int(c1) + right_pad + 1].copy()
    return cropped, (crop_r0, crop_c0)


def _expected_from_metadata(
    shape: tuple[int, int],
    work_items: list[dict],
    direct_colors: dict[int, int],
    synonym_next: dict[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for item in work_items:
        color = _resolve_symbol_index(item["symbol"], direct_colors, synonym_next)
        r0, c0, r1, c1 = item["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _generate_case(with_spec: bool = False):
    symbol_count = random.randint(4, 6)
    symbol_masks = [mask.copy() for mask in random.sample(SYMBOL_LIBRARY, symbol_count)]
    order, direct_colors, synonym_next = _choose_relations(symbol_count)

    extra_work_count = random.randint(0, 2)
    work_symbols = list(range(symbol_count)) + random.choices(range(symbol_count), k=extra_work_count)
    random.shuffle(work_symbols)
    work_rotations = [random.randint(0, 3) for _ in work_symbols]
    if work_symbols and all(rot == 0 for rot in work_rotations):
        work_rotations[random.randrange(len(work_rotations))] = random.randint(1, 3)

    placement = _find_placement(symbol_count, len(work_symbols))
    if placement is None:
        return None
    pair_slots, work_slots = placement

    canvas = np.zeros(MASTER_SHAPE, dtype=int)
    pair_specs = []
    for symbol_idx, left_slot in zip(order, pair_slots):
        left_mask = np.rot90(symbol_masks[symbol_idx], random.randint(0, 3))
        left_patch = _render_symbol(left_mask)
        left_r, left_c = _slot_to_rc(left_slot)
        canvas = paste(canvas, left_patch, left_r, left_c, transparent=BG)

        right_slot = (left_slot[0], left_slot[1] + 1)
        right_r, right_c = _slot_to_rc(right_slot)
        if symbol_idx in direct_colors:
            color = direct_colors[symbol_idx]
            right_patch = _render_square(color)
            right_kind = "color"
            right_value = color
        else:
            target_idx = synonym_next[symbol_idx]
            right_mask = np.rot90(symbol_masks[target_idx], random.randint(0, 3))
            right_patch = _render_symbol(right_mask)
            right_kind = "symbol"
            right_value = target_idx
        canvas = paste(canvas, right_patch, right_r, right_c, transparent=BG)
        pair_specs.append(
            {
                "left_symbol": symbol_idx,
                "left_slot": left_slot,
                "right_kind": right_kind,
                "right_value": right_value,
            }
        )

    work_specs = []
    for symbol_idx, rotation, slot in zip(work_symbols, work_rotations, work_slots):
        patch = _render_symbol(np.rot90(symbol_masks[symbol_idx], rotation))
        r0, c0 = _slot_to_rc(slot)
        canvas = paste(canvas, patch, r0, c0, transparent=BG)
        work_specs.append({"symbol": symbol_idx, "rotation": rotation, "slot": slot})

    grid, origin = _crop_canvas(canvas)
    crop_r0, crop_c0 = origin
    cropped_work_specs = []
    for item in work_specs:
        abs_r0, abs_c0 = _slot_to_rc(item["slot"])
        rel_r0 = abs_r0 - crop_r0
        rel_c0 = abs_c0 - crop_c0
        cropped_work_specs.append(
            {
                **item,
                "bbox": (rel_r0, rel_c0, rel_r0 + BLOCK - 1, rel_c0 + BLOCK - 1),
            }
        )
    expected = _expected_from_metadata(grid.shape, cropped_work_specs, direct_colors, synonym_next)

    if with_spec:
        spec = {
            "symbol_masks": symbol_masks,
            "direct_colors": direct_colors,
            "synonym_next": synonym_next,
            "pair_specs": pair_specs,
            "work_specs": cropped_work_specs,
            "expected": expected,
        }
        return grid, spec
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _generate_case(with_spec=False)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid ARC input")
